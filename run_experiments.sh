# Corrida 1 (Valores por defecto)
python train.py

# Corrida 2 (Mayor regularización)
python train.py --c_param 0.1 --max_iter 200 --solver liblinear

# Corrida 3 (Menor regularización)
python train.py --c_param 10.0 --max_iter 500 --solver newton-cg

# Corrida 4 (Regularización muy fuerte)
python train.py --c_param 0.01 --max_iter 100 --solver lbfgs

# Corrida 5 (Cambio de solver con regularización estándar)
python train.py --c_param 1.0 --max_iter 300 --solver saga
