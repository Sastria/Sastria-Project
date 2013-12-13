'''
Created on Oct 1, 2013

@author: labcontenuti
'''
import socket

HOST = ''                 # local host
PORT = 8889             
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print s
s.bind((HOST, PORT))
#s.setblocking(0)
s.listen(0)
conn, addr = s.accept()
print 'Connected by', addr
while 1:
    data = conn.recv(1024)
    print "data ", data
    if not data: break
    conn.send("Server " + data)
print "closing..."    
conn.close()