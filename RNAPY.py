import random
import numpy as np

class Network(object):       # clase del objeto con argumneto su herencia, es decir,
                          #como primer padre una misma clase: la clase Object.

    def __init__(self, sizes):#se define la función init (como atributo privado) con primer argumento "self" 
                           #(Es una variable especial de sólo lectura que proporciona Python), y
                           #como segundo argumento algún parametro o objeto, en nuestro caso sizes
        self.num_layers = len(sizes) # se toma al objeto num_layer(número de capas) dentro del objeto self
                                     # y se le asigna devolver el número de objetos dentro del objeto sizes
        self.sizes = sizes           #atributo publico
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #list comprehension, esta iterando sobre 
                               #la lista de capas np.random.randn(y, 1) genera matrices con entradas aleatorias
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.epsilon =0.00001
        self.g2 = 0.2
        self.beta = 0.9
        #Los pesos (w) son una matriz que relaciona las flechas entre una capa y otra
        #Esto es para entrenar la red neuronal y poderla usar lo siguiente
    def feedforward(self, a): #Se define feedforward, para evaluar la red neuronal
                           #(self, a(activaciones de la primera capa))
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):   # zip pega los vectores elemento a elemento
            a = sigmoid(np.dot(w, a)+b)
        return a
    def SG(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
#Aquí se definio el Stochastic Gradient Descent, con trainig data como la lista de Tuplas "(x,y)"


     #   """Train the neural network using mini-batch stochastic
      #  gradient descent.  The "training_data" is a list of tuples
       # "(x, y)" representing the training inputs and the desired
        #outputs.  The other non-optional parameters are
        #self-explanatory.  If "test_data" is provided then the
        #network will be evaluated against the test data after each
        #epoch, and partial progress printed out.  This is useful for
        #tracking progress, but slows things down substantially."""
        training_data = list(training_data)
        #if test_data: n_test = len(test_data) #trasformar los datos de entrenamiento en una lista, o tupla
        n = len(training_data)  #El número de datos tenemos para entrenar

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data) #numero de datos de prueba

        for j in range(epochs): #Es para iterar sobre el numero de epocas que se deasea entrenar
            random.shuffle(training_data) # Es como barajar los elementos de la lista 
            mini_batches = [
                training_data[k:k+mini_batch_size]# el tamaño de los mini_batches que se utilixzara 
                                            # para hacer el muestreo
                for k in range(0, n, mini_batch_size)] # donde eta es la taza de aprendizaje
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
            #Si se proporciona el argumento opcional test_data, 
            #entonces el programa evaluará la red después de cada época de entrenamiento e 
            #imprimirá el progreso parcial.
    def update_mini_batch(self, mini_batch, eta): #para cada mini_batch se aplica un único paso de descenso 
                                             #del gradiente.
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]#una vez calculados los gradientes se guardan
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # Aquí x es la entrada de la red, y y es el dato real que queremos que nos 
            #de ademas de actualizar los pesos y sesgos de la red de acuerdo con una única iteración del 
            #descenso de gradiente, utilizando solo los datos de entrenamiento en mini_batch.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # esto nos da el algoritmo de 
            #retropropagación lo que es una forma más eficiente de calcular la funcion de costo
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
            #Se comienza mezclando aleatoriamente los datos de entrenamiento y luego los divide en 
            #minilotes del tamaño apropiado.
    def backprop(self, x, y): #esto nos devuelveuna tupla que representa el
         #gradiente para la función de costo C_x
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]#son listas de capa por capa o elemento a
                                            #elemento de matrices similiraes
        # feedforward
        activation = x
        activations = [x] # genera una lista que alamacena todas las activaciones capa por capa
        zs = [] # hace lo mismo que arriba pero para z 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b  #argumento de la sigmoide. np.dot(w, activation)=wx
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
              sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers): # l variable l es pequeña y l=1 es la ultima capa de la red
            z = zs[-l]  # y -1 la penultima y así sucesivamente, z respresenta la entreda de la capa actual
            sp = sigmoid_prime(z) #es la derivada de la función de activación aplicada a z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #epresenta el error en la capa actual,
                                   # y se calcula utilizando los pesos y el error en la capa siguiente
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #representan los gradientes de los 
                                      #sesgos y pesos
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):#numero de datos que acerto la red
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #Los datos de prueba son una lista de tuplas
        return sum(int(x == y) for (x, y) in test_results)# devuelve l número de entradas de prueba 
                                            #para las cuales la red neuronal genera el resultado correcto.

    def cost_derivative(self, output_activations, y): #derivada de la funcion de costo
        
        return (output_activations-y) #esto nos devuelve el vector de derivadas parciales de la funcion
        #de costo 

#### Miscellaneous functions
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z)) #nos regresa la funcion sigmoidal de la capa  

    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z)) #nos regresa la derivada de la funcion sigmoidal de la capa  