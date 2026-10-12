# Heart Disease — Entrenamiento con MLflow

Script de entrenamiento de un clasificador de **Regresión Logística** para predecir enfermedades cardíacas, con seguimiento completo de experimentos mediante **MLflow**.

---

## Estructura del proyecto

```
Tarea03/
├── train.py              # Script principal de entrenamiento
├── heart.csv             # Dataset (Heart Disease)
├── requirements.txt      # Dependencias Python
├── run_experiments.sh    # Script de runs experimentales
└── mlflow.db             # Base de datos SQLite de MLflow (generada automáticamente)
```

---

## Requisitos

- Python **3.10+**
- Las dependencias están en `requirements.txt`:

| Paquete       | Versión  |
|---------------|----------|
| numpy         | 2.4.4    |
| pandas        | 3.0.2    |
| scikit-learn  | 1.8.0    |
| matplotlib    | 3.10.8   |
| seaborn       | 0.13.2   |
| mlflow        | >= 3.0   |
| skops         | latest   |

---

## Instalación

```bash
# 1. Entrar al directorio del proyecto
cd Tarea03

# 2. (Opcional) Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso básico

```bash
python train.py
```

Al finalizar, el script imprime en consola:

```
Run <run_id> | acc=0.8641 f1=0.8750 recall=0.8889 roc_auc=0.9201
```

---

## Argumentos disponibles

| Argumento             | Tipo    | Defecto                                | Descripción                                               |
|-----------------------|---------|----------------------------------------|-----------------------------------------------------------|
| `--c_param` / `--C`  | float   | `1.0`                                  | Inverso de la fuerza de regularización L2                 |
| `--max_iter`          | int     | `1000`                                 | Iteraciones máximas del solver                            |
| `--solver`            | str     | `lbfgs`                                | Algoritmo: `lbfgs`, `liblinear`, `newton-cg`, `saga`      |
| `--class-weight`      | str     | `none`                                 | Ponderación de clases: `none` o `balanced`                |
| `--imputer-strategy`  | str     | `median`                               | Estrategia de imputación numérica: `median` o `mean`      |
| `--seed`              | int     | `42`                                   | Semilla aleatoria para reproducibilidad                   |
| `--test-size`         | float   | `0.2`                                  | Proporción del conjunto de prueba                         |
| `--data-path`         | str     | `heart.csv`                            | Ruta al archivo CSV del dataset                           |
| `--tracking-uri`      | str     | `sqlite:///mlflow.db`                  | URI de tracking de MLflow                                 |
| `--experiment-name`   | str     | `heart_disease_logreg_hiperparametros` | Nombre del experimento en MLflow                          |
| `--run-name`          | str     | *(automático)*                         | Nombre de la corrida (opcional)                           |
| `--variation`         | str     | `baseline`                             | Etiqueta del factor variado (tag de MLflow)               |

---

## Ejemplos de runs

```bash
# Baseline (valores por defecto)
python train.py

# Mayor regularización (C pequeño)
python train.py --c_param 0.1 --max_iter 200 --solver liblinear --variation mayor_regularizacion

# Menor regularización (C grande)
python train.py --c_param 10.0 --max_iter 500 --solver newton-cg --variation menor_regularizacion

# Regularización muy fuerte
python train.py --c_param 0.01 --max_iter 100 --solver lbfgs --variation regularizacion_muy_fuerte

# Cambio de solver
python train.py --c_param 1.0 --max_iter 300 --solver saga --variation solver_saga

# Clases balanceadas con semilla distinta
python train.py --class-weight balanced --seed 7 --variation class_weight_balanced
```

Para ejecutar todos los runs predefinidos:

```bash
bash run_experiments.sh
```

---

## Pipeline del modelo

```
CSV -> Limpieza -> Train/Test Split (estratificado)
         |
   ColumnTransformer
   |-- Numericas   -> SimpleImputer(median) -> StandardScaler
   |-- Binarias    -> passthrough
   +-- Categoricas -> OneHotEncoder(drop="first")
         |
   LogisticRegression(C, solver, max_iter, class_weight)
```

### Decisiones de limpieza aplicadas automáticamente

- `Cholesterol == 0` y `RestingBP == 0` se convierten a `NaN` (faltantes ocultos).
- Se crea la columna `Cholesterol_missing` (1 si el valor original era 0/NaN) para preservar la señal.
- Se eliminan filas duplicadas.
- El imputador se ajusta **solo con el conjunto de entrenamiento** para evitar data leakage.

---

## Métricas registradas en MLflow

| Métrica                  | Descripción                           |
|--------------------------|---------------------------------------|
| `accuracy`               | Exactitud global                      |
| `precision`              | Precisión                             |
| `recall`                 | Sensibilidad (recall)                 |
| `f1`                     | F1-score                              |
| `roc_auc`                | Área bajo la curva ROC                |
| `accuracy_gain_vs_dummy` | Ganancia sobre clasificador trivial   |
| `tn`, `fp`, `fn`, `tp`   | Componentes de la matriz de confusión |

---

## Artefactos generados por run

- `matriz_confusion.png` — Matriz de confusión sobre el conjunto de prueba
- `curva_roc.png` — Curva ROC sobre el conjunto de prueba
- `metricas.json` — Parámetros y métricas en formato JSON
- `decisiones_limpieza.txt` — Registro de las decisiones de preprocesamiento
- `modelo/` — Modelo serializado con firma MLflow (scikit-learn)

---

## Visualizar experimentos con MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acceder desde el navegador en: `http://127.0.0.1:5000`

---

## Notas

- El dataset esperado es `heart.csv` en el mismo directorio. Se puede cambiar con `--data-path`.
- Cada run queda registrado automáticamente en `mlflow.db` con sus parámetros, métricas y artefactos.
- Si no se especifica `--run-name`, el nombre del run se genera automáticamente con el formato:
  `C{C}_{solver}_it{max_iter}_cw-{class_weight}_seed{seed}`.
