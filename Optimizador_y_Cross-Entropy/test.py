import mnist_loader
import RN # CAMBIÉ EL NOMBRE DEL ARCHIVO NETWORK
import pickle
# COMENTARIO INSERTADO PARA ACTUALIZAR LA DESCRIPCIÓN DEL ARCHIVO EN GITHUB
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=RN.Network([784, 30, 10]) # CAMBIÉ EL NOMBRE DEL ARCHIVO NETWORK
#net.SG_momentum( training_data, 30, 10, 3.0,0.91, test_data=test_data)
net.SG_momentum(training_data, 30, 10 ,3.0, test_data=test_data)
#training_data, epochs, mini_batch_size, eta, momentum,test_data=None)
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo

#archivo_lectura = open("red_prueba.pkl",'rb')
#net = pickle.load(archivo_lectura)
#archivo_lectura.close()

#net.SG( training_data, 10, 50, 0.5, test_data=test_data)

#archivo = open("red_prueba.pkl",'wb')
#pickle.dump(net,archivo)
#archivo.close()
#exit()