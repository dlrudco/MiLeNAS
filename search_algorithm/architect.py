import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, criterion, args):
        gpus = [int(i) for i in args.gpu.split(',')]
        self.is_multi_gpu = True if len(gpus) > 1 else False

        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.criterion = criterion
        self.e2e_latency = 0.

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        self.optimizer = torch.optim.Adam(
            arch_parameters,
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay)
        # self.optimizer = torch.optim.SGD(
        #     arch_parameters,
        #     lr=args.arch_learning_rate)

    # Momentum: https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/
    # V_j = coefficient_momentum * V_j - learning_rate * gradient
    # W_j = V_j + W_jx  x
    # https://www.youtube.com/watch?v=k8fTYJPd3_I
    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        logits = self.model(input)
        loss = self.criterion(logits, target)

        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    # DARTS
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            # logging.info("first order")
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            # logging.info("second order")
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    # ours
    def selector_fn(self, choice_device, min_zero=False, normalize=False):
        #step = self.tanh(self.step)
        step = choice_device
        step.data[step==0] = 1e-3 * torch.randn(step[step==0].shape).to(choice_device.device)#Avoid Divide-By-Zero
        a_step = torch.abs(step.detach())
        out = torch.zeros(step.shape)
        if normalize:
            out = (step-step.detach().min())/(step.detach().max()-step.detach().min())
            return out 
        else:
            out = step/a_step#return 1 for positive, -1 for negative <==> 1 for server side, -1 for device_side
            if min_zero:
                return (out+1)/2 #return 0-1 selection
            else:
                return out # return -1 - 1 selection


    def step_milenas(self, input_train, target_train, input_valid, target_valid, lambda_train_regularizer,
                     lambda_valid_regularizer, trans_layer_num=None, bandwidth=None, gamma=5):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        if trans_layer_num is None:
            loss_train = self.criterion(logits, target_train)
        else:
            loss_train = self.criterion(logits, target_train)
            self.e2e_latency = 0.
            for layer in range(8):
                if layer < trans_layer_num:
                    gamma_ = gamma
                else:
                    gamma_ = 1
                cell = self.model.cells[layer]
                if cell.type=='Edge':
                    if cell.reduction:
                        weight = arch_parameters[1]
                    else:
                        weight = arch_parameters[0]
                    # weight = arch_parameters[2]
                    select = self.selector_fn(arch_parameters[2])
                    channel_select = self.selector_fn(arch_parameters[3], min_zero=True)

                    loss_time, trans_volume = cell.calc_trans_loss(weight, select, channel_select, gamma=5, bandwidth=bandwidth)
                    cell.loss_time = loss_time
                    cell.trans_volume = trans_volume
                    self.e2e_latency += loss_time.item()
                    loss_train += 0.1 * loss_time/(input_train.shape[0]*2)
                    self.e2e_latency += trans_volume.item()/bandwidth
                    loss_train += torch.div(trans_volume, cell.max_trans)/(input_train.shape[0]*bandwidth*2)
                elif cell.type=='Server':
                    # weight = arch_parameters[3]
                    if cell.reduction:
                        weight = arch_parameters[1]
                    else:
                        weight = arch_parameters[0]

                    loss_time, trans_volume = cell.calc_trans_loss(weight)
                    self.e2e_latency += loss_time.item()
                    # loss_train += loss_time/(input_train.shape[0]*2)
                else:
                    if cell.reduction:
                        weight = arch_parameters[1]
                    else:
                        weight = arch_parameters[0]

                    loss_time, trans_volume = cell.calc_trans_loss(weight)
                    self.e2e_latency += gamma_* loss_time.item()


        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters,
            allow_unused=True)

        self.optimizer.zero_grad()

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)

        if trans_layer_num is None:
            loss_val = self.criterion(logits, target_valid)
        else:
            loss_val = self.criterion(logits, target_valid)

            for layer in range(8):
                cell = self.model.cells[layer]
                if layer < trans_layer_num:
                    gamma_ = gamma
                else:
                    gamma_ = 1
                if cell.type=='Edge':
                    if cell.reduction:
                        weight = arch_parameters[1]
                    else:
                        weight = arch_parameters[0]
                    # weight = arch_parameters[2]
                    select = self.selector_fn(arch_parameters[2])
                    channel_select = self.selector_fn(arch_parameters[3], min_zero=True)

                    loss_time, trans_volume = cell.calc_trans_loss(weight, select, channel_select, gamma=5, bandwidth=bandwidth)
                    # self.e2e_latency += loss_time.item()/(input_valid.shape[0]*2)
                    loss_val += 0.1 * loss_time/(input_valid.shape[0]*2)
                    # self.e2e_latency += trans_volume.item()/(input_valid.shape[0]*bandwidth*2)
                    loss_val += 0.1 * torch.div(trans_volume, cell.max_trans)/(input_valid.shape[0]*bandwidth*2)
                elif cell.type=='Server':
                    pass
                    # weight = arch_parameters[3]

                    # loss_time, trans_volume = cell.calc_trans_loss(weight)
                    # self.e2e_latency += loss_time.item()/(input_valid.shape[0]*2)
                    # # loss_train += loss_time/(input_train.shape[0]*2)
                else:
                    pass
                    # if cell.reduction:
                    #     weight = arch_parameters[1]
                    # else:
                    #     weight = arch_parameters[0]

                    # loss_time, trans_volume = cell.calc_trans_loss(weight)
                    # self.e2e_latency += gamma_* loss_time.item()/(input_valid.shape[0]*2)


        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters,
            allow_unused=True)

        # for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
        #     g_val.data.copy_(lambda_valid_regularizer * g_val.data)
        #     g_val.data.add_(g_train.data.mul(lambda_train_regularizer))

        for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        # arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()

        # p_sum = np.sum(
        #     [torch.sum(torch.abs(p)).cpu().detach().numpy() for p in arch_parameters if p.requires_grad])
        # # logging.info("BEFORE step params = %s" % str(p_sum))

        self.optimizer.step()

        # arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        # p_sum = np.sum(
        #     [torch.sum(torch.abs(p)).cpu().detach().numpy() for p in arch_parameters if p.requires_grad])
        # logging.info("AFTER step params = %s" % str(p_sum))

    # ours
    def step_single_level(self, input_train, target_train):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_train_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step_wa(self, input_train, target_train, input_valid, target_valid, lambda_regularizer):
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(loss_train, arch_parameters)

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)
        loss_val = self.criterion(logits, target_valid)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)

        for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
            temp = g_train.data.mul(lambda_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step_AOS(self, input_train, target_train, input_valid, target_valid):
        self.optimizer.zero_grad()
        output_search = self.model(input_valid)
        arch_loss = self.criterion(output_search, target_valid)
        arch_loss.backward()
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)

        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        # calculate w' in equation (7):
        # approximate w(*) by adapting w using only a single training step and enable momentum.
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        logits = unrolled_model(input_valid)
        unrolled_loss = self.criterion(logits, target_valid)
        unrolled_loss.backward()  # w, alpha

        # the first term of equation (7)
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # equation (8) = 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # equation (7)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()

        for v, g in zip(arch_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        named_parameters = self.model.module.named_parameters() if self.is_multi_gpu else self.model.named_parameters()
        for k, v in named_parameters:
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)

        if self.is_multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in model_dict.items():
                logging.info("multi-gpu")
                logging.info("k = %s, v = %s" % (k, v))
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
        else:
            new_state_dict = model_dict

        model_new.load_state_dict(new_state_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        # vector is (gradient of w' on validation dataset)
        R = r / _concat(vector).norm()
        parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)  # w+ in equation (8) # inplace operation

        # get alpha gradient based on w+ in training dataset
        logits = self.model(input)
        loss = self.criterion(logits, target)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_p = torch.autograd.grad(loss, arch_parameters)

        parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        for p, v in zip(parameters, vector):
            p.data.sub_(2 * R, v)  # w- in equation (8)

        # get alpha gradient based on w- in training dataset
        logits = self.model(input)
        loss = self.criterion(logits, target)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        grads_n = torch.autograd.grad(loss, arch_parameters)

        # restore w- to w
        parameters = self.model.module.parameters() if self.is_multi_gpu else self.model.parameters()
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # DARTS
    def step_milenas_2ndorder(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,
                              lambda_train_regularizer, lambda_valid_regularizer):
        self.optimizer.zero_grad()

        # approximate w(*) by adapting w using only a single training step and enable momentum.
        # w has been updated to w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        #print("BEFORE:" + str(unrolled_model.parameters()))

        """(7)"""
        logits_val = unrolled_model(input_valid)
        valid_loss = self.criterion(logits_val, target_valid)
        valid_loss.backward()  # w, alpha

        # the 1st term of equation (7)
        grad_alpha_wrt_val_on_w_prime = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_val_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(grad_w_wrt_val_on_w_prime, input_train, target_train)

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_val_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        grad_alpha_term = unrolled_model.new_arch_parameters()
        for g_new, g in zip(grad_alpha_term, grad_alpha_wrt_val_on_w_prime):
            g_new.data.copy_(g.data)

        """(8)"""
        #unrolled_model_train = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.zero_grad()

        logits_train = unrolled_model(input_train)
        train_loss = self.criterion(logits_train, target_train)
        train_loss.backward()  # w, alpha

        # the 1st term of equation (8)
        grad_alpha_wrt_train_on_w_prime = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_train_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(grad_w_wrt_train_on_w_prime, input_train, target_train)

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_train_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        for g_train, g_val in zip(grad_alpha_wrt_train_on_w_prime, grad_alpha_term):
            # g_val.data.copy_(lambda_valid_regularizer * g_val.data)
            # g_val.data.add_(g_train.data.mul(lambda_train_regularizer))
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        for v, g in zip(arch_parameters, grad_alpha_term):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()
