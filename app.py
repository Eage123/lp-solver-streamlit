import streamlit as st
import pulp
import pandas as pd

st.set_page_config(page_title="Solveur Programmation Linéaire", layout="wide")

st.title("📊 Solveur Générique de Programmation Linéaire ")

# ==============================
# Import Excel
# ==============================
st.subheader("📥 Importer un problème depuis Excel (optionnel)")
excel_file = st.file_uploader("Charger un fichier Excel", type=["xlsx"])

if excel_file:
    df_obj = pd.read_excel(excel_file, sheet_name="objectif")
    df_const = pd.read_excel(excel_file, sheet_name="contraintes")

    nb_vars = len(df_obj)
    nb_constraints = len(df_const)

    obj_coeffs = df_obj["coef"].tolist()

    constraints = []
    for _, row in df_const.iterrows():
        coeffs = row[:-2].tolist()
        sign = row["signe"]
        rhs = row["rhs"]
        constraints.append((coeffs, sign, rhs))

    st.success("Fichier Excel chargé avec succès")("📊 Solveur Générique de Programmation Linéaire")
st.write("Résout n'importe quel problème de programmation linéaire (Max ou Min)")

# ==============================
# Paramètres généraux
# ==============================
col1, col2 = st.columns(2)

with col1:
    nb_vars = st.number_input("Nombre de variables", min_value=1, max_value=20, value=2)

with col2:
    nb_constraints = st.number_input("Nombre de contraintes", min_value=1, max_value=20, value=2)

problem_type = st.selectbox("Type de problème", ["Maximisation", "Minimisation"])

st.divider()

# ==============================
# Fonction objectif
# ==============================
st.subheader("Fonction Objectif")

obj_cols = st.columns(nb_vars)
obj_coeffs = []

for i in range(nb_vars):
    coeff = obj_cols[i].number_input(f"Coeff x{i+1}", value=1.0, key=f"obj_{i}")
    obj_coeffs.append(coeff)

st.divider()

# ==============================
# Contraintes
# ==============================
st.subheader("Contraintes")

constraints = []

for c in range(nb_constraints):
    st.write(f"### Contrainte {c+1}")

    row_cols = st.columns(nb_vars + 2)
    row_coeffs = []

    for i in range(nb_vars):
        coeff = row_cols[i].number_input(
            f"x{i+1}", value=1.0, key=f"c_{c}_{i}"
        )
        row_coeffs.append(coeff)

    sign = row_cols[-2].selectbox(
        "Signe", ["<=", ">=", "="], key=f"sign_{c}"
    )

    rhs = row_cols[-1].number_input(
        "Valeur", value=10.0, key=f"rhs_{c}"
    )

    constraints.append((row_coeffs, sign, rhs))

st.divider()

# ==============================
# Résolution
# ==============================
if st.button("🚀 Résoudre le problème"):

    try:
        # Définition du problème
        if problem_type == "Maximisation":
            prob = pulp.LpProblem("LP_Problem", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("LP_Problem", pulp.LpMinimize)

        # Variables
        variables = [
            pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(nb_vars)
        ]

        # Fonction objectif
        prob += pulp.lpSum(obj_coeffs[i] * variables[i] for i in range(nb_vars))

        # Contraintes
        for row_coeffs, sign, rhs in constraints:
            expr = pulp.lpSum(row_coeffs[i] * variables[i] for i in range(nb_vars))

            if sign == "<=":
                prob += expr <= rhs
            elif sign == ">=":
                prob += expr >= rhs
            else:
                prob += expr == rhs

        # Résolution
        prob.solve()

        st.subheader("✅ Résultats")

        status = pulp.LpStatus[prob.status]
        st.write(f"**Statut :** {status}")

        results = {}
        for v in variables:
            results[v.name] = v.varValue

        results_df = pd.DataFrame(results.items(), columns=["Variable", "Valeur"])
        st.dataframe(results_df)

        st.success(f"Valeur optimale Z = {pulp.value(prob.objective)}")

        # ==============================
        # Graphique (2 variables)
        # ==============================
        if nb_vars == 2:
            import numpy as np
            import matplotlib.pyplot as plt

            x = np.linspace(0, max(v.varValue for v in variables) * 1.5, 400)

            plt.figure()
            for row_coeffs, sign, rhs in constraints:
                if row_coeffs[1] != 0:
                    y = (rhs - row_coeffs[0] * x) / row_coeffs[1]
                    plt.plot(x, y)

            plt.scatter(variables[0].varValue, variables[1].varValue)
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title("Région faisable et solution optimale")
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Erreur : {e}")

st.divider()

st.markdown("""
### 📘 Instructions

1. Choisir le nombre de variables et contraintes
2. Entrer les coefficients de la fonction objectif
3. Remplir chaque contrainte
4. Cliquer sur **Résoudre le problème**


""")
