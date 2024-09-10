#!/usr/bin/env python
# coding: utf-8

# ## Tractatus Quantum-Physicus: Habilidades para Comprender la Mecanica Cuantica
# 
# ##### Capitulo 1.7 
#     Se simula el experimento de 1 y 2 rendijas lanzando particulas clásicas con trayectorias. 
#     A partir del caso 3, al cuantizar las posiciones y velocidades de las particulas, se observan patrones de interferencia.

# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de UNA RENDIJA.   >>>>>>>
#-11111111111111111111 <<<<<<<   CASO 1: Trayectorias rectas CLASICAS. Rango CONTINUO   >>>>>>>


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 



##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 1000000                            # Número de partículas
y_detector = 5                         # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = [-0.1, 0.1]      # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = [-0.1, 0.1]        # Rango de momentos p_x, con masa m = 1
rango_velocidad_y = [1, 10]            # Rango de momentos p_y 
etiqueta = f"""Caso Continuo \n \n x = [-0.1, 0.1] \n    Vx= [-0.1, 0.1]  \n Vy = [1, 10]""" 
titulo_grafico= f"""Experimento de una rendija: Trayectorias Rectas Clásicas""" 


##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en las regiones especificadas
    for _ in range(N):
        x = random.uniform(*rango_x_una_rendija)                                  
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.uniform(*rango_velocidad_x)
        v_y = random.uniform(*rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades



##############################################
####################### Paso 4: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector



##############################################
####################### Paso 5: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(posiciones_iniciales_x, bins=150, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(2, 2, 2)
plt.hist(velocidades_iniciales_x, bins=150, edgecolor='black')
plt.title('Distribución de velocidades iniciales en p_x')
plt.xlabel('velocidades en x')
plt.ylabel('Número de partículas')

plt.subplot(2, 2, 3)
plt.hist(velocidades_iniciales_y, bins=150, edgecolor='black')
plt.title('Distribución de velocidades iniciales en p_y')
plt.xlabel('velocidades en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(2, 2, 4)
plt.hist(posiciones_en_detector, bins=500, edgecolor='blue') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')


plt.tight_layout()
plt.show()


# Generar el histograma de posiciones en el detector
plt.hist(posiciones_en_detector, bins=500, edgecolor='blue') 
plt.title(titulo_grafico, fontsize=14)
plt.xlabel('Posición en x', fontsize=14)
plt.ylabel('Número de partículas', fontsize=14)
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=14,
             horizontalalignment='right', verticalalignment='top')


plt.tight_layout()
plt.show()


# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de UNA RENDIJA.   >>>>>>>
#-22222222222222222222 <<<<<<<   CASO 2: Trayectorias rectas CLASICAS. CONTINUO. COHERENTE.   >>>>>>>


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 



##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 1000000                            # Número de partículas
y_detector = 5                         # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = [-0.1, 0.1]      # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = [-0.1, 0.1]        # Rango de momentos p_x, con masa m = 1
rango_velocidad_y = [5, 5.1]         # Rango de momentos p_y 
etiqueta = f"""Caso Continuo \n y Coherente \n \n x = [-0.1, 0.1] \n    Vx= [-0.1, 0.1]  \n Vy = [5, 5.1]""" 
titulo_grafico= f"""Experimento de una rendija: Trayectorias Rectas Clásicas""" 


##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en las regiones especificadas
    for _ in range(N):
        x = random.uniform(*rango_x_una_rendija)                                  
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.uniform(*rango_velocidad_x)
        v_y = random.uniform(*rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades



##############################################
####################### Paso 4: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector



##############################################
####################### Paso 5: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(posiciones_iniciales_x, bins=150, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(2, 2, 2)
plt.hist(velocidades_iniciales_x, bins=150, edgecolor='black')
plt.title('Distribución de velocidades iniciales en p_x')
plt.xlabel('velocidades en x')
plt.ylabel('Número de partículas')

plt.subplot(2, 2, 3)
plt.hist(velocidades_iniciales_y, bins=150, edgecolor='black')
plt.title('Distribución de velocidades iniciales en p_y')
plt.xlabel('velocidades en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(2, 2, 4)
plt.hist(posiciones_en_detector, bins=500, edgecolor='blue') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')


plt.tight_layout()
plt.show()


# Generar el histograma de posiciones en el detector
plt.hist(posiciones_en_detector, bins=500, edgecolor='blue') 
plt.title(titulo_grafico, fontsize=14)
plt.xlabel('Posición en x', fontsize=14)
plt.ylabel('Número de partículas', fontsize=14)
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=14,
             horizontalalignment='right', verticalalignment='top')


plt.tight_layout()
plt.show()


# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de UNA RENDIJA.   >>>>>>>
#-33333333333333333333 <<<<<<<   CASO 3: CUANTIZADO. Trayectorias rectas CLASICAS  >>>>>>>


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 100000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = np.linspace(-0.01, 0.01, 25)    # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-1., 1., 25)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 25)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso cuantizado \n \n x = (-0.01, 0.01; 25) \n    Vx= (-1, 1; 25)  \n Vy = (10, 11; 25)""" 
titulo_grafico= f"""Experimento de una rendija: Trayectorias Rectas Clásicas""" 





##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en la región especificada
    for _ in range(N):
        x = random.choice(rango_x_una_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()




##############################################
####################### Paso 8: Analizando el experimento: Distribucion por posicion inicial 
######### Se grafica un histograma por cada posicion inicial en x, cada una es en diferente color, y al final coloca cada histograma sobre otro.   
##############################################
# Asignar colores a los valores de posibles_x
colores = plt.cm.jet(np.linspace(0, 1, len(rango_x_una_rendija)))
color_dict = {x: color for x, color in zip(rango_x_una_rendija, colores)}

# Crear un diccionario para agrupar posiciones en el detector por sus colores
posiciones_por_color = {x: [] for x in rango_x_una_rendija}

# Asignar posiciones en el detector a los grupos correspondientes
for pos_detector, pos_inicial in zip(posiciones_en_detector, posiciones):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    posiciones_por_color[clave_mas_cercana].append(pos_detector)

# Graficar histogramas
#plt.figure(figsize=(10, 6))
for x, color in color_dict.items():
    plt.hist(posiciones_por_color[x], bins=500, alpha=0.5, color=color, edgecolor=color, label=f'posibles_x = {x:.2f}')

plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
#plt.legend()
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()






##############################################
####################### Paso 9: Analizando el experimento: Trayectoria por posicion inicial
######### Se grafica las trayectorias de las particulas. Variando el color respecto a la posicion inicial en x.
##############################################
# Graficar los segmentos de recta
plt.figure(figsize=(10, 6))

for pos_inicial, pos_final in zip(posiciones, posiciones_en_detector):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    color = color_dict[clave_mas_cercana]
    
    x_values = [pos_inicial[0], pos_final]
    y_values = [pos_inicial[1], y_detector]
    
    plt.plot(x_values, y_values, color=color, alpha=0.5)

plt.title("Trayectorias de partículas desde la rendija hasta el detector")
plt.xlabel('Posición en x')
plt.ylabel('Posición en y')
plt.annotate(etiqueta, xy=(0.95, 0.2), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.axhline(y=y_detector, color='gray', linestyle='--')
plt.show()# Graficar los segmentos de recta


# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de UNA RENDIJA.   >>>>>>>
#-44444444444444444444 <<<<<<<   CASO 4: CUANTIZADO. Trayectorias rectas CLASICAS   >>>>>>>

##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 100000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = np.linspace(-0.2, 0.2, 10)     # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-1., 1., 7)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 15)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso cuantizado \n \n x = (-0.2, 0.2; 10) \n    Vx= (-1, 1; 7)  \n Vy = (10, 11; 15)""" 
titulo_grafico= f"""Experimento de una rendija: Trayectorias Rectas Clásicas""" 




##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en la región especificada
    for _ in range(N):
        x = random.choice(rango_x_una_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()




##############################################
####################### Paso 8: Analizando el experimento: Distribucion por posicion inicial 
######### Se grafica un histograma por cada posicion inicial en x, cada una es en diferente color, y al final coloca cada histograma sobre otro.   
##############################################
# Asignar colores a los valores de posibles_x
colores = plt.cm.jet(np.linspace(0, 1, len(rango_x_una_rendija)))
color_dict = {x: color for x, color in zip(rango_x_una_rendija, colores)}

# Crear un diccionario para agrupar posiciones en el detector por sus colores
posiciones_por_color = {x: [] for x in rango_x_una_rendija}

# Asignar posiciones en el detector a los grupos correspondientes
for pos_detector, pos_inicial in zip(posiciones_en_detector, posiciones):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    posiciones_por_color[clave_mas_cercana].append(pos_detector)

# Graficar histogramas
#plt.figure(figsize=(10, 6))
for x, color in color_dict.items():
    plt.hist(posiciones_por_color[x], bins=500, alpha=0.5, color=color, edgecolor=color, label=f'posibles_x = {x:.2f}')

plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
#plt.legend()
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()






##############################################
####################### Paso 9: Analizando el experimento: Trayectoria por posicion inicial
######### Se grafica las trayectorias de las particulas. Variando el color respecto a la posicion inicial en x.
##############################################
# Graficar los segmentos de recta
plt.figure(figsize=(10, 6))

for pos_inicial, pos_final in zip(posiciones, posiciones_en_detector):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    color = color_dict[clave_mas_cercana]
    
    x_values = [pos_inicial[0], pos_final]
    y_values = [pos_inicial[1], y_detector]
    
    plt.plot(x_values, y_values, color=color, alpha=0.5)

plt.title("Trayectorias de partículas desde la rendija hasta el detector")
plt.xlabel('Posición en x')
plt.ylabel('Posición en y')
plt.annotate(etiqueta, xy=(0.95, 0.2), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.axhline(y=y_detector, color='gray', linestyle='--')
plt.show()# Graficar los segmentos de recta


# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de UNA RENDIJA.   >>>>>>>
#-55555555555555555555 <<<<<<<   CASO 5: CUANTIZADO. Trayectorias rectas CLASICAS  >>>>>>>


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 10000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = np.linspace(-0.3, 0.3, 3)     # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-1., 1., 7)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 2)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso cuantizado \n \n x = (-0.3, 0.3; 3) \n    Vx= (-1, 1; 7)  \n Vy = (10, 11; 2)""" 
titulo_grafico= f"""Experimento de una rendija: Trayectorias Rectas Clásicas""" 




##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en la región especificada
    for _ in range(N):
        x = random.choice(rango_x_una_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()




##############################################
####################### Paso 8: Analizando el experimento: Distribucion por posicion inicial 
######### Se grafica un histograma por cada posicion inicial en x, cada una es en diferente color, y al final coloca cada histograma sobre otro.   
##############################################
# Asignar colores a los valores de posibles_x
colores = plt.cm.jet(np.linspace(0, 1, len(rango_x_una_rendija)))
color_dict = {x: color for x, color in zip(rango_x_una_rendija, colores)}

# Crear un diccionario para agrupar posiciones en el detector por sus colores
posiciones_por_color = {x: [] for x in rango_x_una_rendija}

# Asignar posiciones en el detector a los grupos correspondientes
for pos_detector, pos_inicial in zip(posiciones_en_detector, posiciones):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    posiciones_por_color[clave_mas_cercana].append(pos_detector)

# Graficar histogramas
#plt.figure(figsize=(10, 6))
for x, color in color_dict.items():
    plt.hist(posiciones_por_color[x], bins=500, alpha=0.5, color=color, edgecolor=color, label=f'posibles_x = {x:.2f}')

plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
#plt.legend()
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()






##############################################
####################### Paso 9: Analizando el experimento: Trayectoria por posicion inicial
######### Se grafica las trayectorias de las particulas. Variando el color respecto a la posicion inicial en x.
##############################################
# Graficar los segmentos de recta
plt.figure(figsize=(10, 6))

for pos_inicial, pos_final in zip(posiciones, posiciones_en_detector):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    color = color_dict[clave_mas_cercana]
    
    x_values = [pos_inicial[0], pos_final]
    y_values = [pos_inicial[1], y_detector]
    
    plt.plot(x_values, y_values, color=color, alpha=0.5)

plt.title("Trayectorias de partículas desde la rendija hasta el detector")
plt.xlabel('Posición en x')
plt.ylabel('Posición en y')
plt.annotate(etiqueta, xy=(0.95, 0.2), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.axhline(y=y_detector, color='gray', linestyle='--')
plt.show()# Graficar los segmentos de recta
plt.figure(figsize=(10, 6))

plt.show()


# In[2]:


#--------------------- <<<<<<<   ExPERIMENTO de 2 Rendijas.   >>>>>>>
#-66666666666666666666 <<<<<<<   CASO 6: Semi-cuantizado: 2 rendijas continuas pero muy pequeñas. Velocidades cuantizadas. Trayectorias rectas clasicas   >>>>>>>


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 1000000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_2_rendijas = np.concatenate(( np.linspace(-1.01, -1.0, 2), np.linspace(1.01, 1.0, 2) ))     # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-15., 15., 50)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 50)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso semi-continuo \n \n 1° rendija = [-1, -1.01]  \n 2° rendija = [1, 1.01]  \n    Vx= (-15, 15; 50)  \n Vy = (10, 11; 50)""" 
titulo_grafico= f"""Experimento de 2 rendijas: Trayectorias Rectas Clásicas""" 




##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_2_rendijas, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []
    
    sub_arrays = np.split(rango_x_2_rendijas, 2)
    rango_x_1er_rendija =sub_arrays[0]
    rango_x_2da_rendija =sub_arrays[1]

    # Generar posiciones en la región especificada
    for _ in range(N):
        
        if random.random() < 0.5:
            x = random.uniform(*rango_x_1er_rendija)
        else:
            x = random.uniform(*rango_x_2da_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_2_rendijas, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de 2 Rendijas.   >>>>>>>
#-77777777777777777777 <<<<<<<   CASO 7: Semi-cuantizado: 2 rendijas cuantizadas en 100mil valores. Velocidades cuantizadas. Trayectorias rectas clasicas   >>>>>>>
#                                        es equivalente a 2 rendijas continuas pero muy pequeñas. 


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 1000000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_2_rendijas = np.concatenate(( np.linspace(-1.01, -1.0, 10000), np.linspace(1.01, 1.0, 10000) ))     # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-15., 15., 50)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 50)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso Cuantizado \n \n 1° r = (-1, -1.01; 10000)\n 2° r = (1, 1.01; 10000)\n    Vx= (-15, 15; 50)\n Vy = (10, 11; 50)""" 
titulo_grafico= f"""Experimento de 2 rendijas: Trayectorias Rectas Clásicas""" 




##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en la región especificada
    for _ in range(N):
        x = random.choice(rango_x_una_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_2_rendijas, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()



# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de 2 Rendijas   >>>>>>>
#-88888888888888888888 <<<<<<<   CASO 8: CUANTIZADO. Trayectorias rectas CLASICAS   >>>>>>>
#                                         Las 2 rendijas son dos puntos infinitesimales fijos


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 200000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = np.concatenate(( np.linspace(-1.01, -1.0, 1), np.linspace(1.01, 1.0, 1) ))     # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-15., 15., 50)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 50)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso cuantizado \n \n 1° rendija = -1.0  \n 2° rendija = 1.0 \n    Vx= (-15, 15; 50)  \n Vy = (10, 11; 50)""" 
titulo_grafico= f"""Experimento de 2 rendijas: Trayectorias Rectas Clásicas""" 




##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en la región especificada
    for _ in range(N):
        x = random.choice(rango_x_una_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()




##############################################
####################### Paso 8: Analizando el experimento: Distribucion por posicion inicial 
######### Se grafica un histograma por cada posicion inicial en x, cada una es en diferente color, y al final coloca cada histograma sobre otro.   
##############################################
# Asignar colores a los valores de posibles_x
colores = plt.cm.jet(np.linspace(0, 1, len(rango_x_una_rendija)))
color_dict = {x: color for x, color in zip(rango_x_una_rendija, colores)}

# Crear un diccionario para agrupar posiciones en el detector por sus colores
posiciones_por_color = {x: [] for x in rango_x_una_rendija}

# Asignar posiciones en el detector a los grupos correspondientes
for pos_detector, pos_inicial in zip(posiciones_en_detector, posiciones):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    posiciones_por_color[clave_mas_cercana].append(pos_detector)

# Graficar histogramas
#plt.figure(figsize=(10, 6))
for x, color in color_dict.items():
    plt.hist(posiciones_por_color[x], bins=500, alpha=0.5, color=color, edgecolor=color, label=f'posibles_x = {x:.2f}')

plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
#plt.legend()
plt.annotate(etiqueta, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()






##############################################
####################### Paso 9: Analizando el experimento: Trayectoria por posicion inicial
######### Se grafica las trayectorias de las particulas. Variando el color respecto a la posicion inicial en x.
##############################################
# Graficar los segmentos de recta
plt.figure(figsize=(10, 6))

for pos_inicial, pos_final in zip(posiciones, posiciones_en_detector):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    color = color_dict[clave_mas_cercana]
    
    x_values = [pos_inicial[0], pos_final]
    y_values = [pos_inicial[1], y_detector]
    
    plt.plot(x_values, y_values, color=color, alpha=0.5)

plt.title("Trayectorias de partículas desde la rendija hasta el detector")
plt.xlabel('Posición en x')
plt.ylabel('Posición en y')
plt.annotate(etiqueta, xy=(0.95, 0.3), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.axhline(y=y_detector, color='gray', linestyle='--')
plt.show()# Graficar los segmentos de recta


# In[1]:


#--------------------- <<<<<<<   ExPERIMENTO de 2 Rendijas   >>>>>>>
#-99999999999999999999 <<<<<<<   CASO 9: CUANTIZADO. Trayectorias rectas CLASICAS   >>>>>>>
#                                         Las 2 rendijas son dos puntos infinitesimales fijos


##############################################
####################### Paso 1: Preparar el Entorno 
##############################################
import numpy as np                # Para manejar funciones matemáticas 
import random                     # Para generar valores aleatorios 
import matplotlib.pyplot as plt   # Para visualizar los resultados 


##############################################
####################### Paso 2: Definir las Condiciones del Experimento  
##############################################
N = 100000  # Número de partículas
y_detector = 5  # Posición del detector en y
y_rendija = 0
rango_x_una_rendija = np.linspace(-0.4, 0.4, 2)     # Rango de posiciones iniciales en x dadas por la rendija
rango_velocidad_x = np.linspace(-1., 1., 60)  # Posibles valores de velocidad en x
rango_velocidad_y = np.linspace(10., 11, 60)  # Posibles valores de velocidad en y
switch_observador ='off'            # prender 'on' o apagar 'off' observador
etiqueta = f"""Caso cuantizado \n \n 1° rendija = -0.4  \n 2° rendija = 0.4  \n    Vx= (-1, 1; 60)  \n Vy = (10, 11; 60)""" 
titulo_grafico= f"""Experimento de 2 rendijas: Trayectorias Rectas Clásicas""" 




##############################################
####################### Paso 3: Generar las posiciones iniciales de las particulas  
##############################################
def generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y):
    posiciones = []
    velocidades = []

    # Generar posiciones en la región especificada
    for _ in range(N):
        x = random.choice(rango_x_una_rendija)
        posiciones.append((x, y_rendija))

        # Generar momentos aleatorios (proporcionales a la velocidad)
        v_x = random.choice(rango_velocidad_x)
        v_y = random.choice(rango_velocidad_y)
        velocidades.append((v_x, v_y))

    return posiciones, velocidades


##############################################
####################### Paso 4: Definir Observador (Le quita Coherencia al sistema)  
##############################################
def observador(posiciones, velocidades, posibles_velocidad_y):
    observaciones = []
    nuevas_velocidades = []
    
    # Cuantizacion actual en Vy, la respetaremos
    step_vy = posibles_velocidad_y[1] - posibles_velocidad_y[0]
    print("cuantizacion en vy: ", step_vy)

    for i, (pos, vel) in enumerate(zip(posiciones, velocidades)):
        x, y = pos
        v_x, v_y = vel

        # Guardar la observación
        observaciones.append((i, x))
       
        
        # Calcular la magnitud inicial del momento
        v0 = np.sqrt(v_x**2 + v_y**2)
        
        
                # Cuantizacion actual en Vy, la respetaremos
        posibles_vy_nuevos = np.arange(0.01, v0 , step_vy)

        
        # Redefinir p_y aleatoriamente entre [0, P0]  # estamos filtrando el caso negativo, de la particula regresandose por la rendija
        v_y_nuevo = random.choice(posibles_vy_nuevos)


        # Redefinir p_x basado en la nueva p_y, y el sentido aleatorio
        v_x_nuevo = np.sqrt(v0**2 - v_y_nuevo**2)*random.choice([-1,1])

        nuevas_velocidades.append((v_x_nuevo, v_y_nuevo))

    return observaciones, nuevas_velocidades


##############################################
####################### Paso 5: Calcular las posiciones cuando chocan con la pantalla   
##############################################
def calcular_posiciones_detector(posiciones, velocidades, y_detector):
    posiciones_en_detector = []

    for (x, y), (v_x, v_y) in zip(posiciones, velocidades):
        # Calcular el tiempo que tarda en llegar a y_detector desde y=0
        tiempo = (y_detector - y) / v_y

        # Calcular la nueva posición en x
        x_en_detector = x + v_x * tiempo
        posiciones_en_detector.append(x_en_detector)

    return posiciones_en_detector


##############################################
####################### Paso 6: Obtener los resultados del experimento   
##############################################
# Generar posiciones y momentos
posiciones, velocidades = generar_particulas(N, rango_x_una_rendija, y_rendija, rango_velocidad_x, rango_velocidad_y)

# Ejecutar el observador
if switch_observador == 'on' :
    print('CON observador')
    observaciones, nuevas_velocidades = observador(posiciones, velocidades, rango_velocidad_y)
else:
    print('SIN observador')
    nuevas_velocidades = velocidades
    

# Calcular las posiciones en el detector
posiciones_en_detector = calcular_posiciones_detector(posiciones, nuevas_velocidades, y_detector)

# Extraer posiciones iniciales y momentos para las gráficas
posiciones_iniciales_x = [pos[0] for pos in posiciones]
velocidades_iniciales_x = [vel[0] for vel in velocidades]
velocidades_iniciales_y = [vel[1] for vel in velocidades]
nuevas_velocidades_x = [vel[0] for vel in nuevas_velocidades]
nuevas_velocidades_y = [vel[1] for vel in nuevas_velocidades]



##############################################
####################### Paso 7: Obtener los resultados del experimento   
##############################################

# Generar histogramas de posiciones iniciales y momentos iniciales
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.hist(posiciones_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de posiciones iniciales en x')
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 2)
plt.hist(velocidades_iniciales_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 3)
plt.hist(velocidades_iniciales_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos iniciales en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')


plt.subplot(3, 2, 4)
plt.hist(nuevas_velocidades_x, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_x')
plt.xlabel('Momento en x')
plt.ylabel('Número de partículas')

plt.subplot(3, 2, 5)
plt.hist(nuevas_velocidades_y, bins=300, edgecolor='black')
plt.title('Distribución de momentos despues de la observacion en p_y')
plt.xlabel('Momento en y')
plt.ylabel('Número de partículas')

# Generar el histograma de posiciones en el detector
plt.subplot(3, 2, 6)
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.65, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()


#plt.figure(figsize=(10, 6))
plt.hist(posiciones_en_detector, bins=500, edgecolor='magenta',color='magenta') 
plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
plt.annotate(etiqueta, xy=(0.65, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()




##############################################
####################### Paso 8: Analizando el experimento: Distribucion por posicion inicial 
######### Se grafica un histograma por cada posicion inicial en x, cada una es en diferente color, y al final coloca cada histograma sobre otro.   
##############################################
# Asignar colores a los valores de posibles_x
colores = plt.cm.jet(np.linspace(0, 1, len(rango_x_una_rendija)))
color_dict = {x: color for x, color in zip(rango_x_una_rendija, colores)}

# Crear un diccionario para agrupar posiciones en el detector por sus colores
posiciones_por_color = {x: [] for x in rango_x_una_rendija}

# Asignar posiciones en el detector a los grupos correspondientes
for pos_detector, pos_inicial in zip(posiciones_en_detector, posiciones):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    posiciones_por_color[clave_mas_cercana].append(pos_detector)

# Graficar histogramas
#plt.figure(figsize=(10, 6))
for x, color in color_dict.items():
    plt.hist(posiciones_por_color[x], bins=500, alpha=0.5, color=color, edgecolor=color, label=f'posibles_x = {x:.2f}')

plt.title(titulo_grafico)
plt.xlabel('Posición en x')
plt.ylabel('Número de partículas')
#plt.legend()
plt.annotate(etiqueta, xy=(0.65, 0.95), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.show()






##############################################
####################### Paso 9: Analizando el experimento: Trayectoria por posicion inicial
######### Se grafica las trayectorias de las particulas. Variando el color respecto a la posicion inicial en x.
##############################################
# Graficar los segmentos de recta
plt.figure(figsize=(10, 6))

for pos_inicial, pos_final in zip(posiciones, posiciones_en_detector):
    valor_x_inicial = pos_inicial[0]
    clave_mas_cercana = min(rango_x_una_rendija, key=lambda x: abs(x - valor_x_inicial))
    color = color_dict[clave_mas_cercana]
    
    x_values = [pos_inicial[0], pos_final]
    y_values = [pos_inicial[1], y_detector]
    
    plt.plot(x_values, y_values, color=color, alpha=0.5)

plt.title("Trayectorias de partículas desde la rendija hasta el detector")
plt.xlabel('Posición en x')
plt.ylabel('Posición en y')
plt.annotate(etiqueta, xy=(0.95, 0.3), xycoords='axes fraction', fontsize=12,
             horizontalalignment='right', verticalalignment='top')

plt.axhline(y=y_detector, color='gray', linestyle='--')
plt.show()# Graficar los segmentos de recta


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




