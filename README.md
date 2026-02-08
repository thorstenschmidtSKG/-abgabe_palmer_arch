# Abgabe_Palmer_Arch

## Penguins — Story-driven Data Visualization

### Dataset
This project uses the **Palmer Penguins dataset**, which contains physical measurements of penguins observed on the Palmer Archipelago in Antarctica.  
Key variables include **bill length**, **bill depth**, species, island, and sex.

---

### Goal of the Visualization
The goal of this visualization is to **communicate how penguin species differ in physical traits**, specifically the **shape of their beaks**.
The main insight is that **penguin species occupy distinct regions in the bill length vs. bill depth space**, making these measurements useful for distinguishing species.

---

### Design Rationale
The visualization follows the principles taught in the course and in *Storytelling with Data*.
- A **single clear message** is emphasized instead of showing all patterns at once  
- **Visual hierarchy** is created using muted context points and one highlighted species  
- Color is used **sparingly and intentionally**, not decoratively  
- Legends are avoided in favor of **direct labeling** to reduce cognitive load  
- Marginal histograms provide **supporting context** without competing for attention  
Interactive elements are included to support exploration but do not replace the primary explanatory message.

---

### How to Use
- The scatter plot shows bill length vs. bill depth  
- One species is highlighted to guide attention  
- Marginal histograms provide distributional context  
- Dashed reference lines indicate where the highlighted species lies relative to the full population  
- Box or lasso selection updates the histograms interactively  

---

### Technical Usage
This project is implemented as a **Streamlit app** using **Bokeh** for interactive visualization.

To run the app locally:

```bash
streamlit run app.py
