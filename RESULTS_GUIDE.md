# Guia de Resultados - SLEAP Training

## 1. Verificar si el job termino

```bash
ssh s4948012@bunya.rcc.uq.edu.au
cd ~/multi-object-computer-vision-tracking

# Ver jobs activos (si no aparece nada, ya termino)
squeue -u $USER

# Ver estado del job especifico
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
```

| Estado | Significado |
|--------|-------------|
| COMPLETED | Termino bien |
| FAILED | Fallo (revisar log) |
| TIMEOUT | Se paso del limite de 6 horas |
| CANCELLED | Lo cancelaste con `scancel` |
| RUNNING | Todavia corriendo |
| PENDING | En cola esperando GPU |

## 2. Donde encontrar los archivos

```
multi-object-computer-vision-tracking/
├── sleap_<jobid>.log                    # Log completo del job
├── results_50ep/                        # Resultados de 50 epochs
│   ├── model/                           # Modelo entrenado (checkpoints)
│   │   ├── best_model.ckpt             # Mejor modelo (menor val loss)
│   │   └── ...
│   └── config_bottomup_ratones_50ep.yaml  # Config usada
├── results_20ep/                        # Resultados de 20 epochs (si lo corriste)
├── data/
│   └── dataset_ratones.slp             # Dataset convertido a SLEAP
└── models/                              # Checkpoints intermedios
```

## 3. Revisar el log

```bash
# Ver todo el log
cat sleap_<jobid>.log

# Ver solo las ultimas lineas (resultado final)
tail -30 sleap_<jobid>.log

# Buscar errores
grep -i "error\|traceback\|failed" sleap_<jobid>.log

# Ver progreso de loss por epoch
grep -i "loss\|epoch" sleap_<jobid>.log
```

## 4. Que medir / que buscar en el log

### Loss (perdida)
- **Training loss**: que tan bien aprende en los datos de entrenamiento
- **Validation loss**: que tan bien generaliza a datos que no vio

Lo que queres ver:
- Ambos loss **bajando** epoch tras epoch
- Que se **estabilicen** hacia el final (significa que convergio)

Senales de problemas:
| Sintoma | Problema | Solucion |
|---------|----------|----------|
| Val loss sube mientras train loss baja | **Overfitting** | Menos epochs, mas augmentation, mas datos |
| Ambos loss se estancan muy alto | **Underfitting** | Mas epochs, mayor learning rate, modelo mas grande |
| Loss oscila mucho | **LR muy alto** | Bajar `--lr` (ej: 0.00005) |
| Loss baja pero muy lento | **LR muy bajo** | Subir `--lr` (ej: 0.0005) |

### Metricas clave
- **Confidence maps loss**: precision de deteccion de keypoints individuales
- **PAF loss (Part Affinity Fields)**: precision de conexion entre keypoints (que parte va con que animal)

## 5. Comparar resultados (20 vs 50 epochs)

```bash
# Ver loss final de cada uno
tail -5 sleap_<jobid_20ep>.log
tail -5 sleap_<jobid_50ep>.log

# Comparar tamanio de resultados
du -sh results_20ep/ results_50ep/
```

Si el loss de 50ep es significativamente menor que 20ep, el modelo se beneficia de mas entrenamiento. Si son parecidos, 20ep es suficiente.

## 6. Usar el modelo entrenado

### Predecir poses en un video nuevo
```bash
source .venv/bin/activate
sleap-nn predict video.mp4 --model results_50ep/model/
```

### Predecir en un directorio de imagenes
```bash
sleap-nn predict imagenes/ --model results_50ep/model/
```

### Exportar predicciones
Las predicciones se guardan en formato `.slp` que se puede abrir con SLEAP GUI o procesar con `sleap-io` en Python:

```python
import sleap_io as sio

labels = sio.load_file("predictions.slp")
for lf in labels:
    print(f"Frame {lf.frame_idx}: {len(lf.instances)} animales detectados")
    for inst in lf.instances:
        for node, point in zip(labels.skeletons[0].nodes, inst.points):
            print(f"  {node.name}: ({point.x:.1f}, {point.y:.1f})")
```

## 7. Que hacer si falla

| Error | Causa | Solucion |
|-------|-------|----------|
| `No such file or directory` | Path incorrecto | Verificar `cd` en submit_bunya.sh |
| `ModuleNotFoundError` | Falta dependencia en .venv | `pip install <modulo>` |
| `CUDA out of memory` | GPU sin memoria | Bajar `--batch_size` (ej: 2) |
| `TIMEOUT` | Training muy largo | Pedir mas tiempo `--time=12:00:00` o menos epochs |
| `Disk quota exceeded` | Sin espacio en disco | `export TMPDIR=~/tmp && mkdir -p $TMPDIR` |

## 8. Siguiente paso: reentrenar con mas epochs

Si el loss seguia bajando al final de 50 epochs:

```bash
bash train.sh 100
```

O ajustar hiperparametros editando la linea en `submit_bunya.sh`:
```bash
python train_sleap.py \
    --epochs 100 \
    --batch_size 4 \       # bajar a 2 si hay OOM
    --lr 0.0001 \          # subir/bajar segun comportamiento del loss
    --output_dir "results_100ep"
```
