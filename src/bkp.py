
'''
    with st.expander('Mapa de Vendas'):
        coluna1, coluna2 = st.columns(2)
        vendas = ss.groupby('State')['Sales'].sum().reset_index()
        fig = px.choropleth(
            vendas,
            locations='State',
            locationmode='USA-states',
            color='Sales'
        )
        fig.update_layout(title='Vendas', geo = dict(
                scope='usa',
                projection=go.layout.geo.Projection(type = 'albers usa'),
                showlakes=True, # lakes
                lakecolor='rgb(255, 255, 255)'),
        )

        coluna1.plotly_chart(fig)
        lucros = ss.groupby('Country')['Profit'].sum().reset_index()
        fig = px.choropleth(
            lucros,
            locations='Country',
            locationmode='country names',
            color='Profit'
        )
        fig.update_layout(title='Lucro',template="plotly_white")
        coluna2.plotly_chart(fig)

            fig = px.choropleth(
                out_pai,
                locations='referencia',
                locationmode='USA-states',
                color='outlier'
            )
            fig.update_layout(title='Outlier', geo=dict(
                scope='usa',
                projection=go.layout.geo.Projection(type='albers usa'),
                showlakes=True,  # lakes
                lakecolor='rgb(255, 255, 255)'),
            )
            st.plotly_chart(fig)

'''