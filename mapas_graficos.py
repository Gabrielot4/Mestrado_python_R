import pandas as pd

br_cities = pd.read_excel('operacoes_judicializacoes.xlsx', sheet_name='Sheet3')
print(br_cities)

import plotly.express as px


### MAPA COM NÚEMRO DE OPERAÇÕES POR ESTADO
fig_oper = px.scatter_mapbox(br_cities, lat="LAT", lon="LONG", hover_name="Capital", hover_data=["Estado", "Oper"],
                        color_discrete_sequence=["green"], zoom=3, height=1000, width=1000, size="Oper", size_max=50, labels="Estado")
fig_oper.update_layout(mapbox_style="open-street-map")
fig_oper.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig_oper.show()

### MAPA COM NÚEMRO DE JUDICIALIZAÇÕES POR ESTADO
fig_jud = px.scatter_mapbox(br_cities, lat="LAT", lon="LONG", hover_name="Capital", hover_data=["Estado", "Jud"],
                        color_discrete_sequence=["green"], zoom=3, height=1000, width=1000, size="Jud", size_max=50, labels="Estado")
fig_jud.update_layout(mapbox_style="open-street-map")
fig_jud.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig_jud.show()

### MAPA COM PROPORÇÃO PADRONIZADA ENTRE OPERAÇÕES E JUDICIALIZAÇÕES POR ESTADO
fig_padron = px.scatter_mapbox(br_cities, lat="LAT", lon="LONG", hover_name="Capital", hover_data=["Estado", "Prop"],
                        color_discrete_sequence=["green"], zoom=3, height=1000, width=1000, size="Prop", size_max=50, labels="Estado")
fig_padron.update_layout(mapbox_style="open-street-map")
fig_padron.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig_padron.show()