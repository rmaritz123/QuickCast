
import pandas as pd
from io import BytesIO

def generate_sku_report(sku, df, kpi_dict, copilot_text):
    output = BytesIO()
    kpi_df = pd.DataFrame(kpi_dict)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Forecast", index=False)
        kpi_df.to_excel(writer, sheet_name="KPIs", index=False)
        writer.sheets["KPIs"].write("H2", "Copilot Insight")
        writer.sheets["KPIs"].write("H3", copilot_text)
    output.seek(0)
    return output
