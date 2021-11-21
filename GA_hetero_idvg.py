# -*- coding: utf-8 -*-


import tensorflow as tf
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import time
from pandas.core.frame import DataFrame
import copy
seed_value = 1
os.environ['PYTHONASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)  
df = pd.read_csv('Die11-idvg-tot.csv')
column = [0,1,2,3,4,5,6,7]
df.columns= column
#normalization
w = df.iloc[:,0:1] #width
l = df.iloc[:,1:2] #length
g = df.iloc[:,2:3] #It is actually Vd in Die11-idvg.csv
d = df.iloc[:,3:4] #It is Vg in Die11-idvg.csv
y = df.iloc[:,5:6]# Log Id
w =np.array(w, dtype=np.float32)
l =np.array(l, dtype=np.float32)
g =np.array(g, dtype=np.float32)
d =np.array(d, dtype=np.float32)
scalerw = MinMaxScaler()
w_minmax =scalerw.fit_transform(w)
scalerl = MinMaxScaler()
l_minmax =scalerl.fit_transform(l)
scalerg = MinMaxScaler()
g_minmax =scalerg.fit_transform(g)
scalerd = MinMaxScaler()
d_minmax =scalerd.fit_transform(d)
scalery = MinMaxScaler()
y_minmax =scalery.fit_transform(y)

rand = np.random.permutation(y.shape[0])

Y = y_minmax[rand]
L = l_minmax[rand]
W = w_minmax[rand]
G = g_minmax[rand]
D =d_minmax[rand]

split_percentage = 0.9
L_train = L[:int(l.shape[0]*split_percentage)]
L_test = L[int(l.shape[0]*split_percentage):]
W_train = W[:int(l.shape[0]*split_percentage)]
W_test = W[int(l.shape[0]*split_percentage):]
G_train = G[:int(l.shape[0]*split_percentage)]
G_test = G[int(l.shape[0]*split_percentage):]
D_train = D[:int(l.shape[0]*split_percentage)]
D_test = D[int(l.shape[0]*split_percentage):]
X_train = [W_train,L_train,G_train,D_train]
X_test = [W_test,L_test,G_test,D_test]   
Y_train = Y[:int(l.shape[0]*split_percentage)]
Y_test = Y[int(l.shape[0]*split_percentage):]




#define how many layers and blocks. also give them ids
def create_blocks(layer_max,block_max):
    tot_l = random.randint(2, layer_max)
    layers = []    
    for i in range(tot_l):
        blocks = random.randint(1, block_max)
        layers.append(blocks)
    k = 1
    blocks_id = []
    for i in layers:
        ids = []
        for n in range(i):    
            ids.append((k,n+1))       
        k+=1
        blocks_id.append(ids) 
    return blocks_id


#define the connection(blocks in i layer are only allowed to connect to blocks in i-1 layer),and
#randomly assign number of neuron and activation function of each block
def create_conn(n_max,blocks_id):
    Hidden = {}
    Output = {}
    input_key = [(-1,1),(-1,2),(-1,3),(-1,4)]
    activations = ['sigmoid','tanh']
    for key, value in enumerate(blocks_id):
        for i in value:
            if key == 0:
                r = random.randint(3,4)   
                conn_1 = random.sample(input_key,r)
                conn_1.sort()
                act = random.choice(activations)
                neurons = random.randint(1, n_max)
                Hidden[i] = [neurons, act,conn_1]
            else:
                l = len(blocks_id[key-1])
                r2 = random.randint(1,l)
                conn = random.sample(blocks_id[key-1],r2)
                conn.sort()
                act = random.choice(activations)
                neurons = random.randint(1, n_max)
                Hidden[i] = [neurons, act,conn]
    Output = {}
    out_blocks_l = len(blocks_id[-1])
    r_out = random.randint(out_blocks_l//2+1,out_blocks_l)
    out_conn = random.sample(blocks_id[-1],r_out)
    out_conn.sort()
    Output[(0,0)]=out_conn
    return Hidden, Output

def create_genome(l_max,b_max,n_max):
    ids = create_blocks(l_max,b_max)
    hidden,output =  create_conn(n_max,ids)
    return [hidden,output]

 #Crossover operator, input parent genomes, return two child genomes. Crossover point for two genomes 
 #may be different.  
def crossover(genome1,genome2): 
    hidden_g1 = copy.deepcopy(genome1[0])
    hidden_g2 = copy.deepcopy(genome2[0])
    output_g1 = copy.deepcopy(genome1[1])
    output_g2 = copy.deepcopy(genome2[1])
    k1 = list(hidden_g1.keys())
    k1.sort()
    k2= list(hidden_g2.keys())
    k2.sort()
    layer_g1 = k1[-1][0]
    layer_g2 = k2[-1][0]
    crosspoint_g1 = random.randint(1,layer_g1-1)
    crosspoint_g2 = random.randint(1,layer_g2-1)
    newhidden1_1 = {}
    newhidden1_2 = {}
    newhidden2_1 = {}
    newhidden2_2 = {}
    
    for key in k1:
        if key[0]<=crosspoint_g1:
            newhidden1_1[key] = hidden_g1[key]
        else:
            newhidden1_2[key] = hidden_g1[key]
    for key in k2:
        if key[0]<=crosspoint_g2:
            newhidden2_1[key] = hidden_g2[key]
        else:
            newhidden2_2[key] = hidden_g2[key]
     
    newhidden1_1.update(newhidden2_2)
    newhidden2_1.update(newhidden1_2)
    newg1 = [newhidden1_1,output_g2]
    newg2 = [newhidden2_1,output_g1]
    return  newg1,newg2


# Build the model using Keras functional api. If the input genome has missing connection ids or
# reapeted block ids, create_model function will repair the connection and return a keras model.
def create_model(genome): #include mutation
    Value = {}
    Value[(-1,1)] = tf.keras.Input(shape=(None,1),name = 'W')
    Value[(-1,2)]= tf.keras.Input(shape=(None,1),name = 'L')
    Value[(-1,3)]= tf.keras.Input(shape=(None,1),name = 'G')
    Value[(-1,4)]= tf.keras.Input(shape=(None,1),name = 'D')
    hidden = genome[0]
    output = genome[1]
    layers = []
    keys = list(hidden.keys())
    keys.sort()
    for key in keys:
        layers.append(key[0])
    layers=set(layers)
    layer_ids = [] 
    for i in layers:
        L = []
        for k in keys:
            if k[0]==i:
                L.append(k)
        layer_ids.append(L)  
    
    for layer, ids in enumerate(layer_ids):
        if layer ==0:
            for i in ids:
                input_ids_value = []
                neuron = hidden[i][0]
                activation = hidden[i][1]
                for conn in hidden[i][2]:
                    input_ids_value.append(Value[conn])
                inputdata = tf.keras.layers.Concatenate()(input_ids_value)
                l = tf.keras.layers.Dense(neuron,activation)(inputdata)
                Value[i]=l  

        else:
            for i in ids:
                input_ids_value = []
                neuron = hidden[i][0]
                activation = hidden[i][1]
                conns = hidden[i][2]
                for key, conn in enumerate(conns):
                    if conn not in layer_ids[layer-1]:
                        new_conn = random.choice(layer_ids[layer-1])
                        conns[key]=new_conn
                conns=list(set(conns))
                conns.sort()
                hidden[i][2] = conns 
                for conn in hidden[i][2]:
                    input_ids_value.append(Value[conn])
                if len(input_ids_value) == 1:
                    l = tf.keras.layers.Dense(neuron,activation)(input_ids_value[0])
                else:
                    inputdata = tf.keras.layers.Concatenate()(input_ids_value)
                    l = tf.keras.layers.Dense(neuron,activation)(inputdata)
                Value[i]=l      
           
    output_conn = output[(0,0)]
    for key,conn in enumerate(output_conn):
        if conn not in layer_ids[-1]:
            new_conn = random.choice(layer_ids[-1])
            output_conn[key]=new_conn
    output_conn = list(set(output_conn))
    output_conn.sort()
    output[(0,0)]=output_conn
    
    output_ids_value=[]
    for ids in output[(0,0)]:
        output_ids_value.append(Value[ids])
    if len(output_ids_value) == 1:
        outdata = output_ids_value[0]
        out = tf.keras.layers.Dense(1)(outdata)
    else:
        outdata = tf.keras.layers.Concatenate()(output_ids_value)
        out =tf.keras.layers.Dense(1)(outdata)
    
    model_input = [Value[(-1,1)],Value[(-1,2)],Value[(-1,3)],Value[(-1,4)]]
    model = tf.keras.models.Model(inputs=model_input, outputs=out)
    return model

# The validation loss of model training is set to be each model's fitness score. 
def fitness(model,x_train,y_train):
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    my_callbacks= [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
    model.compile(optimizer =opt, loss = 'mean_squared_error', metrics = ['acc'])
    history = model.fit(x_train,y_train, epochs= 1000,batch_size = 64,validation_split = 0.2, callbacks=my_callbacks)
    val_loss = history.history['val_loss']
    # loss = history.history['loss']
    fitness = val_loss[-1]
    epochs =len(val_loss)
    return fitness,epochs


class Individual:
    def __init__(self,genome):
        self.genome = genome
        self.model = create_model(self.genome)
        self.fitness,self.epochs = fitness(self.model,X_train,Y_train)
        self.params = self.model.count_params()
        
        
def init_population(pop_size,l_max,b_max,n_max):
    Population_0 = []
    for i in range(pop_size):
        genome_ini = create_genome(l_max,b_max,n_max)
        Population_0.append(Individual(genome_ini))
    return Population_0

#tournament selection ->return better genome(only 1)
def selection(pop): 
        s = random.sample(pop, 2)
        if s[0].fitness<s[1].fitness:
            best = s[0]
        else:
            best = s[1]
        return best
    
#N:population size, input population, return mate_pool for reproduction
def sel_parents(pop,N): 
    mate_pool=[]
    for i in range(N//2):        
        parent1 = selection(pop)
        parent2 = selection(pop)
        mate_pool.append([parent1,parent2])
    return mate_pool

def offspring(mate_pool):
    child_population = []
    for parents in mate_pool:
        g1 = parents[0].genome
        g2 = parents[1].genome
        childg1, childg2 = crossover(g1,g2)
        child1=Individual(childg1)
        child2=Individual(childg2)
        child_population.append(child1)
        child_population.append(child2)   
    return child_population


#Environment selection. Inviduals with better fitness score are allowed to survive, others will be 
#eliminated so that the population will remain the same in each generation.
def env_select(pop,pop_size):
    pop.sort(key = lambda x: x.fitness)
    pop=pop[0:pop_size]
    return pop

#Evaluate the whole population
def evaluation(pop):
    evol_results = []
    evol_params=[]
    evol_epochs = []
    for x in pop:
        evol_results.append(x.fitness)
        evol_params.append(x.params)
        evol_epochs.append(x.epochs)
    return [evol_results,evol_params,evol_epochs]

#Run the evolution and return the fittest individual
def run_evolution(gen_limit,pop_size,layer_max,block_max,neuron_max): 
    generation = {}
    results = []
    epochs = []
    params = []
    pop = init_population(pop_size,layer_max,block_max,neuron_max)
    k = 0
    generation[k] = pop
    eva = evaluation(pop)
    results.append(eva[0])
    params.append(eva[1])
    epochs.append(eva[2])
    for i in range(gen_limit):
        k+=1
        mate_pool = sel_parents(pop,pop_size)
        child_pop = offspring(mate_pool)
        pop = child_pop+pop
        pop = env_select(pop,pop_size)
        generation[k] = pop
        eva = evaluation(pop)
        results.append(eva[0])
        params.append(eva[1])
        epochs.append(eva[2])
        i+=1

    best_individual = pop[0]
    return best_individual,results, generation,params,epochs




#%%
start = time.process_time()      
best,results,gens,params,epoch = run_evolution(25,20,5,4,6)
best_model = best.model
best_genome = best.genome
fitnessscore = best.fitness
stop = time.process_time()  
processtime = stop-start
best_model.summary()
best_model.save('models/GA_logid_hetero_546.h5')
df_evol_results= DataFrame(results)
df_evol_results.to_csv("results/fitness_logid_546.csv", index = False)
df_evol_params= DataFrame(params)
df_evol_params.to_csv("results/params_logid_546.csv", index = False)
df_evol_epochs= DataFrame(epoch)
df_evol_epochs.to_csv("results/epochs_logid_546.csv", index = False)


