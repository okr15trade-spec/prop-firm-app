# Probabilidad de pasar el reto (Prop Firm)

Aplicación Streamlit para comparar historiales de operaciones en CSV y estimar la probabilidad de alcanzar el objetivo antes de los límites de pérdida, mediante remuestreo Monte Carlo.

## Ejecutar

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

La estimación es una aproximación: el modelo remuestrea operaciones individuales y considera un día nuevo cada cinco operaciones. Comprueba las reglas exactas de la firma antes de interpretar los resultados.
