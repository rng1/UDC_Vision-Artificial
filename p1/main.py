import skimage
import histograms
import morphological
import filters
import edge

circles = "p1/img/circles.png"
circles1 = "p1/img/circles1.png"
jota = "p1/img/jota.png"
lena = "p1/img/lena.jpg"
morph = "p1/img/morph.png"
grays = "p1/img/grays.png"
grid = "p1/img/grid.png"
splash = "p1/img/image.png"
image2 = "p1/img/image2.png"
grid0 = "p1/img/grid0.png"
eq0 = "p1/img/eq0.png"

"""
>> histograms.plot_output(in_image)
"""
#histograms.plot_output(skimage.io.imread(eq0))

"""
>> filters.plot_output(in_image, mode="all", filter_size=9, sigma=9)

mode (por defecto, "all")
       all : Muestra todas las imágenes en un mismo panel.
     split : Muestra todas las imágenes individualmente.
    median : Muestra solo el filtro de medianas.
     gauss : Muestra solo el filtro de gauss.
    
filter (por defecto, 9)
    Tamaño del kernel a usar en el filtro de medianas
    
sigma (por defecto, 9)
    Desviación típica a la hora de construir el kernel de gauss
"""
#filters.plot_output(skimage.io.imread(grid0),
#                    mode="median",
#                    filter_size=5,
#                    sigma=9)

"""
>> morphological.plot_output(in_image, mode="all")

mode (por defecto, "all")
       all : Muestra todas las operaciones en distintos paneles.
    dil_er : Muestra las operaciones de dilatación y erosión.
        op : Muestra la operación de apertura.
        cl : Muestra la operación de cierre.
       h_m : Muestra la operación de hit-or-miss.
"""
morphological.plot_output(skimage.io.imread(morph),
                          mode="dil_er")

"""
>> edge.plot_output(in_image, mode="all", operator="sobel", sigma_LoG=2, sigma_canny=1.5, tlow=0.3, thigh=0.5)

mode (por defecto, "all")
              all : Muestra todas las imágenes en un mismo panel.
            split : Muestra todas las imágenes individualmente.
         gradient : Muestra solo las gradientes seleccionadas.
    all_gradients : Muestra todos las gradientes en un mismo panel.
              log : Muestra solo el filtro Laplaciano de Gaussiano (LoG).
            canny : Muestra el resultado final del algoritmo Canny.
      canny_steps : Muestra todos los pasos del algoritmo Canny en un mismo panel.
      
operator (por defecto, "sobel")
    roberts
    prewitt
    sobel
    central_diff
    
sigma_LoG (por defecto, 2)
    Desviación típica a la hora de realizar el LoG.
    
sigma_canny (por defecto, 1.5)
    Desviación típica a la hora de realizar el algoritmo de Canny.
    
sigma_canny (por defecto, 1.5)
    Desviación típica a la hora de realizar el algoritmo de Canny.
    
tlow (por defecto, 0.3)
    Umbral de histéresis bajo del algoritmo de Canny.
    
thigh (por defecto, 0.5)
    Umbral de histéresis alto del algoritmo de Canny.
"""
 #edge.plot_output(skimage.io.imread(grid0),
 #                 mode="log",
 #                 operator="sobel",
 #                 sigma_LoG=1,
 #                 sigma_canny=5,
 #                 tlow=0.2,
 #                 thigh=0.7)
