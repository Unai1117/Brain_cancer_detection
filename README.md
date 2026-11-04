# Clasificación de tumores cerebrales en resonancia magnética

Se clasifican cortes 2D de RM cerebral en cuatro clases (glioma, meningioma, pituitario y sin tumor) con el [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) de Kaggle. El trabajo compara PCA con clasificadores clásicos, una CNN desde cero, una ResNet18 preentrenada con tres grados de ajuste y el efecto del aumento de datos. Antes de entrenar se limpia el dataset (copias exactas, casi-copias y pares con etiquetas contradictorias) y se construye una validación agrupada por similitud perceptual (pHash) para evitar fugas, ya que el dataset no incluye identificadores de paciente.

La explicación completa está en la memoria (`Memoria_RFA.pdf`); este README solo describe el repositorio y cómo reproducirlo.

## Resultados

F1 macro, media de 3 semillas (E1 se entrena una sola vez). El *test limpio* excluye las imágenes de test con una casi-copia (pHash d ≤ 6) en entrenamiento o validación.

| Experimento | Modelo | Val | Test | Test limpio |
|---|---|---|---|---|
| E1 | PCA (K=500) + QDA | 0,802 | 0,750 | 0,633 |
| E3a | ResNet18, *linear probing* | 0,895 | 0,825 | 0,723 |
| E2 | CNN desde cero | 0,955 | 0,907 | 0,809 |
| E3c | ResNet18, `layer4` + cabeza | 0,959 | 0,911 | 0,814 |
| E4 | CNN desde cero + aumento | 0,970 | 0,918 | 0,827 |
| E3b | ResNet18, *fine-tuning* completo | 0,981 | 0,927 | 0,838 |
| E4 | ResNet18 `layer4` + aumento | 0,975 | 0,934 | 0,852 |
| **E4** | **ResNet18 completo + aumento** | **0,989** | **0,944** | **0,873** |

La caída entre validación y test se concentra en 62 gliomas de test que no son de 512 × 512 (todos los gliomas de train y val lo son). Sin ellos, el mejor modelo obtiene un F1 de 0,995 en test y acierta el 99,4-99,5 % de las 1117 imágenes restantes. Este análisis es posterior a la evaluación en test y no se usó para modificar ningún modelo (Sección 9 de la memoria).

## Contenido

| Fichero | Descripción |
|---|---|
| `rfa_trabajo.ipynb` | Cuaderno principal: EDA y limpieza, partición, E1-E4, evaluación en test y análisis de errores |
| `rfa_test_sin62.ipynb` | Análisis a posteriori: evaluación en test sin los 62 gliomas de otro formato (Tabla 7 de la memoria) |
| `Memoria_RFA.pdf` | Memoria del trabajo |
| `Clasificación_de_tumores_cerebrales_en_RM___Defensa_RFA.pdf` | Presentación de la defensa |

## Datos

El dataset no se incluye en el repositorio. Puede descargarse desde la web de Kaggle o con la CLI:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data
```

El cuaderno espera esta estructura, relativa al directorio desde el que se ejecuta:

```
data/
├── Training/{glioma,meningioma,notumor,pituitary}/
└── Testing/{glioma,meningioma,notumor,pituitary}/
```

El trabajo usa la versión con 7200 imágenes (5600 Training / 1600 Testing, 1400 y 400 por clase). La segunda celda de código debe imprimir `7200 {'Training': 5600, 'Testing': 1600}`; si no, la versión descargada es otra y ni la limpieza ni los resultados coincidirán con los de la memoria.

## Entorno

Python 3 con:

```
torch torchvision timm torchinfo scikit-learn scipy numpy pandas matplotlib pillow jupyter
```

Los pesos de ImageNet de la ResNet18 los descarga `timm` desde Hugging Face Hub la primera vez. Los experimentos se ejecutaron en una GPU CUDA; el código también funciona en MPS o CPU, pero las redes serán mucho más lentas. Los `DataLoader` usan `num_workers=6`; redúcelo si la máquina tiene menos núcleos.

## Reproducción

1. Ejecutar `rfa_trabajo.ipynb` de principio a fin desde el directorio que contiene `data/`. La limpieza genera `splits.csv`, que leen todos los experimentos posteriores, y cada experimento guarda sus pesos e historiales en `results/`. Con la GPU usada, el entrenamiento de las 21 redes (7 configuraciones × 3 semillas) lleva algo más de una hora.
2. Ejecutar `rfa_test_sin62.ipynb`, que reutiliza `splits.csv` y los pesos de `results/`. Su primera comprobación verifica que el test completo reproduce exactamente `results/evaluacion_test.csv`; si falla, algo no coincide con el cuaderno principal. Con `HACER_E1 = False` se omite E1 y el PCA.

La partición es determinista (`StratifiedGroupKFold` con `random_state=42`) y cada red usa la semilla `42 + s`, con `s ∈ {0, 1, 2}`. Al no forzar algoritmos deterministas en cuDNN, los resultados de las redes pueden variar ligeramente entre ejecuciones o máquinas. Las parejas de imágenes que se muestran al elegir el umbral de pHash se sortean sin semilla, así que esas figuras cambian en cada ejecución.

### Ficheros generados

| Ruta | Contenido |
|---|---|
| `splits.csv` | Una fila por imagen tras la limpieza: ruta, partición original, clase, MD5, pHash, `split` (`train`/`val`/`test`) y `casi_copia` (imagen de test con casi-copia en train+val) |
| `cache_{train,val,test}_224.npz` | Imágenes en gris a 224 × 224 para E1. Se borran al inicio de E1 porque dependen de la partición |
| `results/{e2,e3,e3_l4,e3_entero,e4,e4_l4,e4_entero}_semilla{0,1,2}.pt` | Pesos de la última época de cada red y semilla, los que se evalúan en test |
| `results/*.csv` | Historiales de entrenamiento por época y semilla |
| `results/evaluacion_test.csv` | Métricas en test completo y limpio por modelo y semilla |
| `results/evaluacion_test_sin62.csv` | Métricas por subconjunto de test (generado por `rfa_test_sin62.ipynb`) |

Las ejecuciones exploratorias con una sola semilla guardan además `results/e3_entero.pt`, `results/e3_layer4.pt`, `results/e3_todo.csv` y `results/e3_layer4.csv`, que no intervienen en la evaluación final.

## Protocolo

Se mantiene la partición oficial de Kaggle: `Testing` es el test y `Training` se divide en entrenamiento y validación agrupando las casi-copias (componentes conexas a pHash d ≤ 6), de modo que nunca quedan a ambos lados. Todas las redes se entrenan 60 épocas con entropía cruzada, AdamW (lr = 1e-3), planificador coseno y batch de 32. El modelo final es el de la última época, no el mejor en validación, y el test se usa una sola vez con todos los modelos ya fijados.

## Limitaciones

La agrupación por pHash solo aproxima una partición por paciente. Se usa un único fold de validación, con el que también se eligieron los hiperparámetros, así que las cifras de validación son algo optimistas. La limpieza eliminó del test las imágenes con copia idéntica en `Training`, por lo que los resultados no son directamente comparables con trabajos que usan las 1600 imágenes de test originales.

## Autor

Unai López Lozano de Arnilla
