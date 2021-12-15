f = open("./FP2.txt", "r")
loss=[]
counter=0
for x in f:
  counter+=1
  lines = x.split('Loss: ')
  print("(",counter,",",lines[1].rstrip(),")") 
