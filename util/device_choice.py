import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableChoice(torch.nn.Module):
	def __init__(self, shape, debug=False):
		super(DifferentiableChoice, self).__init__()
		self.shape = shape
		self.step = nn.Parameter(torch.randn(shape), requires_grad=True)
		#self.tanh = nn.Tanh()
		if debug:
			print("Initial")
			print(self.step)

	def forward(self, x):
		#step = self.tanh(self.step)
		step = self.step
		a_step = torch.abs(step.detach())
		out = torch.zeros(self.shape)
		out[step==0] = 1e-3 * torch.randn(shape)[step==0]#Avoid Divide-By-Zero
		out = step/a_step#return 1 for positive, -1 for negative
		return out

	def choice(self, target1, target2, weight=[1, 1]):
		choice = self.forward()
		# choose target1 for choice value -1, choose target2 for choice value 1
		choose1 = (1 - choice)/2
		choose2 = (choice - 1)/2
		out = weight[0]  * torch.mul(choose1, target1) +\
			weight[1]  * torch.mul(choose2, target2)
		return out


def main():
	shape = (1,)
	ss = DifferentiableChoice(debug=False, shape=shape)
	x = torch.randn(shape)
	target1 = torch.randint(1, 20, shape)
	target2 = torch.randint(1, 20, shape)
	#print(target1, target2)
	optimal = torch.min(torch.cat((target1, target2), dim=0),dim=0)

	optimizer = torch.optim.SGD(ss.parameters(), lr=0.05)

	for i in range(100):
		optimizer.zero_grad()
		out = ss(x)
		loss1 = torch.sum(torch.mul(target1, out+1))
		loss2 = torch.sum(torch.mul(target2, 1-out))
		loss = loss1 + loss2
		loss.backward()
		optimizer.step()
		# Expected Final parameter = [0., 1., 0., 1., 0., 1.]
	#print(loss.item()/2)#, torch.abs(out-1), out<0, out > 0)
	print(loss.item()/2 == torch.sum(optimal[0]).item())#, optimal[1])
	return loss.item()/2 == torch.sum(optimal[0]).item()
if __name__ == "__main__":
	acc = 0
	for _ in range(100):
		a = main()
		acc += a
	print(acc)