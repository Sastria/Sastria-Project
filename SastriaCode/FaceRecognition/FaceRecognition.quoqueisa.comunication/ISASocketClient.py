'''
Created on Oct 1, 2013

@author: labcontenuti
'''
import socket

HOST = ''    # The remote host
PORT = 8889               
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.send(100)

while 1:
    data = s.recv(1024)
    print 'Received', repr(data)
s.close()
#print "close"