import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchprof

from search_space.genotypes import Genotype, Genotype_Edge
from search_space.genotypes import PRIMITIVES
from search_space.operations import OPS, FactorizedReduce, ReLUConvBN
from search_space.utils import count_parameters_in_MB
from util.device_choice import DifferentiableChoice

import time
# class MixedOp(nn.Module):

#     def __init__(self, C, stride):
#         super(MixedOp, self).__init__()
#         self._ops = nn.ModuleList()
#         for primitive in PRIMITIVES:
#             op = OPS[primitive](C, stride, False)
#             if 'pool' in primitive:
#                 op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
#             self._ops.append(op)

#     def forward(self, x, weights):
#         # w is the operation mixing weights. see equation 2 in the original paper.
#         return sum(w * op(x) for w, op in zip(weights, self._ops))
class MixedOp(nn.Module):

    def __init__(self, C, stride, trans=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._profiles = {}
        self.trans=trans
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._profiles[op.__str__()] = {'Out_Size':-1, 'In_Size':-1, 'Exec_Time':-1}
        self.handlers = []
        self.paths = [("ModuleLise", "0"), ("ModuleLise", "1"),
        ("ModuleLise", "2"),("ModuleLise", "3"),("ModuleLise", "4"),
        ("ModuleLise", "5"),("ModuleLise", "6"),("ModuleLise", "7"),]
        self._register_hook()
        self.hook_count = 0


    def _get_features_hook(self, module, input, output):
        module_name = module.__str__()
        if module_name == 'Zero()':
            self._profiles[module_name]['Out_Size'] = 0
            self._profiles[module_name]['In_Size'] = 0
        else:
            self._profiles[module_name]['Out_Size'] = output.element_size() * output.nelement()
            self._profiles[module_name]['In_Size'] = input[0].element_size() * input[0].nelement()
        #print("feature MEM size:{}".format(self._profiles[module_name]['Out_Size']))

    def _register_hook(self):
        for i, module in enumerate(self._ops):
            self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def moving_average(self, original, new, count):
        return (original*count + new)/(count+1)

    def parse_time(self, str):
        if 'us' in str:
            cpu_time = 1e-3 * float(str.replace('us','').replace(' ',''))
        elif 'ms' in str:
            cpu_time = float(str.replace('ms','').replace(' ',''))
        else:
            return None
        return cpu_time


    def parse_prof(self, prof):
        result = prof.__str__()
        key = None
        new_time = {}
        for line in result.split('\n'):
            if line == '':
                continue
            if line.split('|')[0][:3] == '├──' or line.split('|')[0][:3] == '└──':
                key = line.split('|')[0].replace(line.split('|')[0][:3],'').replace(' ','')
                module_name = self._ops[int(key)].__str__()
                new_time[module_name] = {}
                new_time[module_name]['Exec_Time'] = 0.
                cpu_total = line.split('|')[2]
                cpu_time = self.parse_time(cpu_total)
                if cpu_time is None:
                    continue
                new_time[module_name]['Exec_Time'] = cpu_time
            else:
                if key is None:
                    continue
                else:
                    cpu_total = line.split('|')[2]
                    cpu_time = self.parse_time(cpu_total)
                if cpu_time is None:
                    continue
                new_time[module_name]['Exec_Time'] = cpu_time + new_time[module_name]['Exec_Time']
        for modules in self._profiles:
            if modules == 'Zero()':
                self._profiles[modules]['Exec_Time'] = 0.
            else:
                self._profiles[modules]['Exec_Time'] = self.moving_average(
                    self._profiles[modules]['Exec_Time'], new_time[modules]['Exec_Time'],
                    self.hook_count)


    def forward(self, x, weights):
        # w is the operation mixing weights. see equation 2 in the original paper.
        if self.hook_count < 25:
            with torchprof.Profile(self._ops, use_cuda=True, profile_memory=True) as prof:
                out = sum(w * op(x) for w, op in zip(weights, self._ops))
            self.parse_prof(prof)
            self.hook_count += 1
        elif self.hook_count == 25:
            self.remove_handlers()
            out = sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            out = sum(w * op(x) for w, op in zip(weights, self._ops))

        return out


class EdgeCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(EdgeCell, self).__init__()
        self.type='Edge'
        self.loss_time = 0.
        self.trans_volume=0.
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        
        self.C = C
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def _define_dependency(self):
        self.calc_dependency=[
        [],[],[],[],[0,1],[],[],[0,1],[0,1,2,3,4],
        [],[],[0,1], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8]]
        self.data_dependency = []

    def forward(self, s0, s1, weights, channel_selects):
        s0 = torch.mul(channel_selects[0].view(1,self.C,1,1),self.preprocess0(s0))
        s1 = torch.mul(channel_selects[1].view(1,self.C,1,1),self.preprocess1(s1))
        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(torch.mul(channel_selects[i+2].view(1,self.C,1,1),s))

        return torch.cat(states[-self._multiplier:], dim=1)

    def WeightedRuntime(self, profiles, weights):
        edge_time = torch.tensor(0.).cuda()
        softmax_weights = F.softmax(weights)
        for module_idx, module in enumerate(profiles):
            exec_time = profiles[module]['Exec_Time'] * softmax_weights[module_idx]
            edge_time += exec_time
        return exec_time


    def WeightedTransSize(self, profiles, weights, io):
        trans_size = torch.tensor(0.).cuda()
        softmax_weights = F.softmax(weights)
        for module_idx, module in enumerate(profiles):
            memsize = profiles[module][io+'_Size'] * softmax_weights[module_idx]
            trans_size += memsize
        return trans_size


    def set_device_for_state(self, states, selection):
        preds = ['s0', 's1']
        ignore = []#Override 'E' if in this list --> node s_k is already on 'S' so edge cannot be 'E'
        for i in range(self._steps):
            curr_state = states[f's{i+2}']
            for pred in preds:
                if states[pred]['device'] == 'S':
                    curr_state['device'] = 'S'
            
            if curr_state['device'] == 'E':
                for in_edge in curr_state['efrom']:
                    if selection[in_edge] == 1:
                        curr_state['device'] = 'S'
            
            if curr_state['device'] == 'S':
                ignore.extend(curr_state['eto'])

            states[f's{i+2}'] = curr_state
            preds.append(f's{i+2}')
        return states, ignore


    def calc_edge_time(self, states, selection, edges, weights, channel_selection,
     ignore, gamma, bandwidth):
        trans_candidate = {'s0':{'Data': 0., 'Count': 0}, 
        's1':{'Data': 0., 'Count': 0}, 
        's2':{'Data': 0., 'Count': 0}, 
        's3':{'Data': 0., 'Count': 0}, 
        's4':{'Data': 0., 'Count': 0}, 
        's5':{'Data': 0., 'Count': 0}
        }
        total_edge_time = torch.tensor(0.).cuda()
        total_trans_volume = torch.tensor(0.).cuda()
        for idx, edge in enumerate(edges):

            cs_idx = int(edge['from'][1])
            beta = torch.sum(channel_selection[cs_idx])/self.C #Reduce execution time according to channel selection
            runtime = beta*self.WeightedRuntime(self._ops[idx]._profiles, weights[idx])

            if states[edge['from']]['device'] == states[edge['to']]['device']: #No Transmission since two node on same side
                if states[edge['from']]['device'] == 'E':
                    if selection[idx] == 1:
                        total_edge_time += runtime
                    elif selection[idx] == -1.:
                        total_edge_time += gamma*(1-selection[idx])*runtime/2
                    else:
                        print('Wrong selection value')
                        breakpoint()
                elif states[edge['from']]['device'] == 'S':
                    if selection[idx] == 1:
                        total_edge_time += (selection[idx]+1)*runtime/2
                    elif selection[idx] == -1.:
                        total_edge_time += gamma*runtime
                    else:
                        print('Wrong selection value')
                        breakpoint()
                else:
                    print("Wrong device setting")
                    breakpoint()
            elif states[edge['from']]['device'] == 'E':
                total_edge_time += (selection[idx]+1)*runtime/2
                total_edge_time += gamma*(1-selection[idx])*runtime/2
                if selection[idx] == 1:
                    trans_candidate[edge['from']]['Data'] += beta*selection[idx]*self.WeightedTransSize(self._ops[idx]._profiles, weights[idx], 'In')
                    trans_candidate[edge['from']]['Count'] += 1
                elif selection[idx] == -1:
                    cs_idx_ = int(edge['to'][1])
                    beta_ = torch.sum(channel_selection[cs_idx_])/self.C
                    trans_candidate[edge['to']]['Data'] += beta_*(-selection[idx])*self.WeightedTransSize(self._ops[idx]._profiles, weights[idx], 'Out')
                    trans_candidate[edge['to']]['Count'] += 1
                else:
                    print('Wrong selection value')
                    breakpoint()
            else:
                print("Former node should be on Edge not Server")
                breakpoint()
                raise AssertionError
        # print(trans_candidate)
        for cs_idx, finals in enumerate(['s2','s3','s4','s5']):
            beta = torch.sum(channel_selection[cs_idx+2])/self.C
            if states[finals]['device'] == 'E' and trans_candidate[finals]['Count'] == 0:
                op_name = list(self._ops[states[finals]['efrom'][0]]._profiles.keys())[-1]
                trans_candidate[finals]['Data'] += beta*self._ops[states[finals]['efrom'][0]]._profiles[op_name]['Out_Size']
                trans_candidate[finals]['Count'] += 1

        for candidate in trans_candidate:
            if trans_candidate[candidate]['Count'] != 0:
                total_trans_volume += trans_candidate[candidate]['Data']/(trans_candidate[candidate]['Count']*bandwidth)
        # print(trans_candidate)
        # print(total_edge_time, total_trans_volume)
        return total_edge_time, total_trans_volume

    def calc_trans_loss(self, weights, selection, channel_selection, gamma=5, bandwidth=None):
        edge_dependency=[
        [],[],[],[],[0,1],[],[],[0,1],[0,1,2,3,4],
        [],[],[0,1], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8]]

        states = {'s0': {'from':[], 'to':['s2','s3','s4','s5'],'efrom':[],'eto':[0, 2, 5, 9], 'device':'E'},
            's1':{'from':[], 'to':['s2','s3','s4','s5'],'efrom':[],'eto':[1, 3, 6, 10], 'device':'E'},
            's2':{'from':['s0','s1'], 'to':['s3','s4','s5'],'efrom':[0, 1],'eto':[4, 7, 11], 'device':'E'},
            's3':{'from':['s0','s1','s2'], 'to':['s4','s5'],'efrom':[2, 3, 4],'eto':[8, 12], 'device':'E'},
            's4':{'from':['s0','s1','s2','s3'], 'to':['s5'],'efrom':[5, 6, 7, 8],'eto':[13], 'device':'E'},
            's5':{'from':['s0','s1','s2','s3','s4'], 'to':[],'efrom':[9, 10, 11, 12, 13],'eto':[], 'device':'E'}}

        edges = [
        {'from': 's0', 'to' : 's2'},#0
        {'from': 's1', 'to' : 's2'},#1
        {'from': 's0', 'to' : 's3'},#2
        {'from': 's1', 'to' : 's3'},#3
        {'from': 's2', 'to' : 's3'},#4
        {'from': 's0', 'to' : 's4'},#5
        {'from': 's1', 'to' : 's4'},#6
        {'from': 's2', 'to' : 's4'},#7
        {'from': 's3', 'to' : 's4'},#8
        {'from': 's0', 'to' : 's5'},#9
        {'from': 's1', 'to' : 's5'},#10
        {'from': 's2', 'to' : 's5'},#11
        {'from': 's3', 'to' : 's5'},#12
        {'from': 's4', 'to' : 's5'}]#13

        states, ignore = self.set_device_for_state(states, selection)
        edge_time = self.calc_edge_time(states, selection, edges, weights, channel_selection,
            ignore, gamma, bandwidth)
        return edge_time


class ServerCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(ServerCell, self).__init__()
        self.type='Server'
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s0)
        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

    def calc_edge_time(self, edges, weights):
        total_edge_time = torch.tensor(0.).cuda()
        total_trans_volume = torch.tensor(0.).cuda()
        for idx, edge in enumerate(edges):
            runtime = self.WeightedRuntime(self._ops[idx]._profiles, weights[idx])
            total_edge_time += runtime
        return total_edge_time, total_trans_volume

    def calc_trans_loss(self, weights):
        edges = [
        {'from': 's0', 'to' : 's2'},#0
        {'from': 's1', 'to' : 's2'},#1
        {'from': 's0', 'to' : 's3'},#2
        {'from': 's1', 'to' : 's3'},#3
        {'from': 's2', 'to' : 's3'},#4
        {'from': 's0', 'to' : 's4'},#5
        {'from': 's1', 'to' : 's4'},#6
        {'from': 's2', 'to' : 's4'},#7
        {'from': 's3', 'to' : 's4'},#8
        {'from': 's0', 'to' : 's5'},#9
        {'from': 's1', 'to' : 's5'},#10
        {'from': 's2', 'to' : 's5'},#11
        {'from': 's3', 'to' : 's5'},#12
        {'from': 's4', 'to' : 's5'}]#13
        edge_time = self.calc_edge_time(edges, weights)
        return edge_time


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.type='Normal'
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

    def calc_edge_time(self, edges, weights):
        total_edge_time = torch.tensor(0.).cuda()
        total_trans_volume = torch.tensor(0.).cuda()
        for idx, edge in enumerate(edges):
            runtime = self.WeightedRuntime(self._ops[idx]._profiles, weights[idx])
            total_edge_time += runtime
        return total_edge_time, total_trans_volume

    def calc_trans_loss(self, weights):
        edges = [
        {'from': 's0', 'to' : 's2'},#0
        {'from': 's1', 'to' : 's2'},#1
        {'from': 's0', 'to' : 's3'},#2
        {'from': 's1', 'to' : 's3'},#3
        {'from': 's2', 'to' : 's3'},#4
        {'from': 's0', 'to' : 's4'},#5
        {'from': 's1', 'to' : 's4'},#6
        {'from': 's2', 'to' : 's4'},#7
        {'from': 's3', 'to' : 's4'},#8
        {'from': 's0', 'to' : 's5'},#9
        {'from': 's1', 'to' : 's5'},#10
        {'from': 's2', 'to' : 's5'},#11
        {'from': 's3', 'to' : 's5'},#12
        {'from': 's4', 'to' : 's5'}]#13
        edge_time = self.calc_edge_time(edges, weights)
        return edge_time


class InnerCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, weights):
        super(InnerCell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        # len(self._ops)=2+3+4+5=14
        offset = 0
        keys = list(OPS.keys())
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                weight = weights.data[offset + j]
                choice = keys[weight.argmax()]
                op = OPS[choice](C, stride, False)
                if 'pool' in choice:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self._ops.append(op)
            offset += i + 2

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)



class ModelForModelSizeMeasure(nn.Module):
    """
    This class is used only for calculating the size of the generated model.
    The choices of opeartions are made using the current alpha value of the DARTS model.
    The main difference between this model and DARTS model are the following:
        1. The __init__ takes one more parameter "alphas_normal" and "alphas_reduce"
        2. The new Cell module is rewriten to contain the functionality of both Cell and MixedOp
        3. To be more specific, MixedOp is replaced with a fixed choice of operation based on
            the argmax(alpha_values)
        4. The new Cell class is redefined as an Inner Class. The name is the same, so please be
            very careful when you change the code later
        5.

    """

    def __init__(self, C, num_classes, layers, criterion, alphas_normal, alphas_reduce,
                 steps=4, multiplier=4, stem_multiplier=3):
        super(ModelForModelSizeMeasure, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                cell = InnerCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                 alphas_reduce)
            else:
                reduction = False
                cell = InnerCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                                 alphas_normal)

            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input_data):
        s0 = s1 = self.stem(input_data)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, 
        steps=4, multiplier=4, stem_multiplier=3, trans_layer_num=None):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self.trans_layer_num = trans_layer_num

        C_curr = stem_multiplier * C  # 3*16
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # for layers = 8, when layer_i = 2, 5, the cell is reduction cell.
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if i == trans_layer_num:
                self.trans_C = C_curr
                cell = EdgeCell(steps, multiplier, 
                    C_prev_prev, C_prev, C_curr, 
                    reduction, reduction_prev)
            elif i == trans_layer_num + 1:
                cell = ServerCell(steps, multiplier, 
                    C_prev_prev, C_prev, C_curr, 
                    reduction, reduction_prev)
            else: 
                cell = Cell(steps, multiplier, 
                    C_prev_prev, C_prev, C_curr, 
                    reduction, reduction_prev)
            reduction_prev = reduction

            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def selector_fn(self, choice_device, min_zero=False):
        #step = self.tanh(self.step)
        step = choice_device
        step[step==0] = step[step==0] + 1e-3 * torch.randn(step[step==0].shape).to(choice_device.device)#Avoid Divide-By-Zero
        a_step = torch.abs(step.detach())
        out = torch.zeros(step.shape)
        out = step/a_step#return 1 for positive, -1 for negative <==> 1 for server side, -1 for device_side
        if min_zero:
            return (out+1)/2 #return 0-1 selection
        else:
            return out # return -1 - 1 selection

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i == self.trans_layer_num:
                weights = F.softmax(self.alphas_edge, dim=-1)
                channel_weights = self.selector_fn(self.select_channel, min_zero=True)
                s0, s1 = s1, cell(s0, s1, weights, channel_weights)
                continue
            elif i == self.trans_layer_num + 1:
                weights = F.softmax(self.alphas_server, dim=-1)
            else: 
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)

            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_edge = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_server = nn.Parameter(1e-3 * torch.randn(k, num_ops)) if self.trans_layer_num < 7 else self.alphas_normal
        self.select_device = nn.Parameter(torch.randn(k))
        self.select_channel = nn.Parameter(torch.ones(self._steps+2,self.trans_C))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.alphas_edge,
            self.alphas_server,
            self.select_device,
            self.select_channel
        ]

    def new_arch_parameters(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        alphas_edge = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        alphas_server = nn.Parameter(1e-3 * torch.randn(k, num_ops)) if self.trans_layer_num < 7 else self.alphas_normal
        select_device = nn.Parameter(torch.randn(k))
        select_channel = nn.Parameter(torch.ones(self._steps+2,self.trans_C))
        _arch_parameters = [
            alphas_normal,
            alphas_reduce,
            alphas_edge,
            alphas_server,
            select_device,
            select_channel
        ]
        return _arch_parameters

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _isCNNStructure(k_best):
            return k_best >= 4

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            cnn_structure_count = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k

                    if _isCNNStructure(k_best):
                        cnn_structure_count += 1
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene, cnn_structure_count

        with torch.no_grad():
            gene_normal, cnn_structure_count_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce, cnn_structure_count_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
            gene_edge, cnn_structure_count_edge = _parse(F.softmax(self.alphas_edge, dim=-1).data.cpu().numpy())
            gene_server, cnn_structure_count_server = _parse(F.softmax(self.alphas_server, dim=-1).data.cpu().numpy())
            concat = range(2 + self._steps - self._multiplier, self._steps + 2)
            genotype = Genotype_Edge(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat,
                edge=gene_edge, edge_concat=concat,
                server = gene_server, server_concat=concat
            )
        return genotype, cnn_structure_count_normal, cnn_structure_count_reduce, cnn_structure_count_edge, cnn_structure_count_server

    def get_current_model_size(self):
        model = ModelForModelSizeMeasure(self._C, self._num_classes, self._layers, self._criterion,
                                         self.alphas_normal, self.alphas_reduce, self._steps,
                                         self._multiplier, self._stem_multiplier)
        size = count_parameters_in_MB(model)
        # This need to be further checked with cuda stuff
        del model
        return size
