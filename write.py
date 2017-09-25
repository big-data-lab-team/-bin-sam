from hdfs import InsecureClient
from hdfs import Client

# /user/gao/

data = "00000"
client = InsecureClient("http://localhost:50070", user="gao")
# client = Client(url= "http://localhost:50070", root='/')
client.write('/c.txt', data=data, encoding='utf-8', append=True)


# with client.write('a.txt', encoding='utf-8') as writer:
#     writer.write(data)