if __name__ == "__main__":
	print("Executed when invoked directly")
	testModel = Net1()
	testModel(torch.randn(1,1,28,28))
else:
	print("Executed when imported")