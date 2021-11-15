## Instrucciones


1. Crear un repo git en donde se bajen esto

```
git init
git remote add origin <nuestra-nueva-url-de-git>
```

2. Bajarse los repositorios de `pybind` y `eigen` como submódulos

```
git submodule init
git submodule add https://github.com/eigenteam/eigen-git-mirror
git submodule add https://github.com/pybind/pybind11
git mv eigen-git-mirror eigen
# Elegimos versiones de eigen y pybind
cd pybind11/ && git checkout v2.2.4 && cd ..
cd eigen && git checkout 3.3.7 && cd ..
```

3. Instalar requerimientos (*Previamente activar el entorno virtual. Ver  más abajo*)

```
pip install -r requirements.txt
```

4. Descomprimir datos

```

5. Correr Jupyter

```
jupyter lab
```

Listo. Ya pueden disfrutar del TP3

### Datos


### Otros directorios

En `src/` está el código de C++, en particular en `src/metnum.cpp` está el entry-point de pybind.

En `notebooks/` hay ejemplos para correr partes del TP usando sklearn y usando la implementación en C++.

## Creación de un entorno virtual de python

### Con pyenv

```
curl https://pyenv.run | bash
```

Luego, se sugiere agregar unas líneas al bashrc. Hacer eso, **REINICIAR LA CONSOLA** y luego...

```
pyenv install 3.6.5
pyenv global 3.6.5
pyenv virtualenv 3.6.5 tp2
```

En el directorio del proyecto

```
pyenv activate tp3
```

### Directamente con python3
```
python3 -m venv tp3
source tp3/bin/activate
```

### Con Conda
```
conda create --name tp3 python=3.6.5
conda activate tp3
```

## Instalación de las depencias
```
pip install -r requirements.txt
```

## Correr notebooks de jupyter

```
cd notebooks
jupyter lab
```
o  notebook
```
jupyter notebook
```


## Compilación
Ejecutar la primera celda del notebook `cm.ipynb`


