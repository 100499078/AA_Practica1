import streamlit as st
import pandas as pd
import joblib
 
def transform_pdays(df):
    """Transforma la columna pdays: crea wascontacted y pdays_binned."""
    df = df.copy()
    df["wascontacted"] = df["pdays"].apply(lambda x: 0 if x == -1 else 1)
    df["pdays_binned"] = df["pdays"].apply(
        lambda x: "no_contacto" if x == -1
        else "0_99"      if x < 100
        else "100_199"   if x < 200
        else "200_399"   if x < 400
        else "400_plus"
    )
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("notebooks/modelo_final.joblib")
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo del modelo en 'notebooks/modelo_final.joblib'. Verifica la ruta.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

model = load_model()
if model is None:
    st.stop()

if "historial" not in st.session_state:
    st.session_state.historial = []

st.title("Predicción de Suscripción a Depósito Bancario")
st.markdown(
    "Introduce los datos del cliente para predecir si suscribirá un depósito a plazo."
)
 
st.header("Datos del cliente")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    st.subheader("Perfil personal")
    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    job = st.selectbox("Tipo de trabajo", [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ])
    marital = st.selectbox("Estado civil", ["married", "single", "divorced"])
    education = st.selectbox("Educación", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("¿Crédito impagado?", ["no", "yes"])
 
with col2:
    st.subheader("Situación financiera")
    balance = st.number_input("Balance anual medio (€)", value=1000, step=100,
    help="Puede ser negativo si el cliente tiene deudas.")
    housing = st.selectbox("¿Tiene hipoteca?", ["yes", "no"])
    loan = st.selectbox("¿Tiene préstamo personal?", ["no", "yes"])
 
with col3:
    st.subheader("Último contacto")
    contact = st.selectbox("Tipo de contacto", ["cellular", "telephone", "unknown"])
    day = st.number_input("Día del mes", min_value=1, max_value=31, value=15)
    month = st.selectbox("Mes", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    duration = st.number_input("Duración de la llamada (segundos)",min_value=0, value=180, step=10,
    help="Duración del último contacto en segundos. Un valor de 0 indica que no hubo llamada.")
 
st.header("Campaña de marketing")
col4, col5 = st.columns(2)
 
with col4:
    campaign = st.number_input("Nº contactos en esta campaña", min_value=1, value=1,
    help="Número de veces que se ha contactado al cliente durante esta campaña.")
    pdays = st.number_input("Días desde último contacto en campaña anterior (-1 = sin contacto)", min_value=-1, value=-1,
    help="Introduce -1 si el cliente no fue contactado en ninguna campaña anterior.")
 
with col5:
    previous = st.number_input("Nº contactos en campañas anteriores", min_value=0, value=0)
    poutcome = st.selectbox(
        "Resultado campaña anterior", ["unknown", "failure", "success", "other"]
    )

if st.button("🔍 Predecir"):
    if duration == 0:
        st.warning("⚠️ Duración 0 segundos: no hubo contacto real. La predicción puede no ser fiable.")
    
    if campaign > 20:
        st.warning("⚠️ Número de contactos en campaña inusualmente alto. Verifica el dato.")

    wascontacted = 0 if pdays == -1 else 1

    pdays_binned = (
        "no_contacto" if pdays == -1
        else "0_99"    if pdays < 100
        else "100_199" if pdays < 200
        else "200_399" if pdays < 400
        else "400_plus"
    )
 
    input_data = pd.DataFrame([{
        "age":          age,
        "job":          job,
        "marital":      marital,
        "education":    education,
        "default":      default,
        "balance":      balance,
        "housing":      housing,
        "loan":         loan,
        "contact":      contact,
        "day":          day,
        "month":        month,
        "duration":     duration,
        "campaign":     campaign,
        "pdays":        pdays,
        "previous":     previous,
        "poutcome":     poutcome,
    }])
 
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None
 
    st.divider()
    if prediction == "yes":
        st.success("✅ El cliente **SÍ** suscribirá el depósito.")
    else:
        st.error("❌ El cliente **NO** suscribirá el depósito.")
 
    if proba is not None:
        classes = model.classes_
        proba_dict = dict(zip(classes, proba))
        
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            st.metric("Probabilidad de suscripción", f"{proba_dict['yes']:.1%}")
            st.progress(proba_dict['yes'])
        
        with col_no:
            st.metric("Probabilidad de no suscripción", f"{proba_dict['no']:.1%}")
            st.progress(proba_dict['no'])

    registro = input_data.copy()
    registro["prediccion"] = "Sí ✅" if prediction == "yes" else "No ❌"
    if proba is not None:
        registro["prob_suscripcion"] = f"{proba_dict['yes']:.1%}"
    st.session_state.historial.append(registro)
 
    with st.expander("Ver datos introducidos"):
        st.dataframe(input_data)

if st.session_state.historial:
    st.divider()
    st.header("Historial de predicciones de esta sesión")
    historial_df = pd.concat(st.session_state.historial, ignore_index=True)
    st.dataframe(historial_df, use_container_width=True)

    st.download_button(
        label="Descargar historial como CSV",
        data=historial_df.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_clientes.csv",
        mime="text/csv"
    )

    if st.button("Limpiar historial"):
        st.session_state.historial = []
        st.rerun()
 