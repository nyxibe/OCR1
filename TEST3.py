
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from random import *
from PIL import Image as img

def intro():
    import streamlit as st

    st.write("# Welcome PROJET 7 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

    @st.cache
    def from_data_file(filename):
        url = (
            "http://raw.githubusercontent.com/streamlit/"
            "example-data/master/hello/v1/%s" % filename
        )
        return pd.read_json(url)

    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={
                        "latitude": 37.76,
                        "longitude": -122.4,
                        "zoom": 11,
                        "pitch": 50,
                    },
                    layers=selected_layers,
                )
            )
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def nyx_demo():
    
     ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit NYX 0001 </h1>
    </div>    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)  

    
    from datetime import datetime
    def convert(date_time) :
        f='%Y-%m-%d'
        datetime_str = datetime.strptime(date_time,f)
        return datetime_str
    
    import matplotlib.pyplot as plt
    def boxplotNyx2 (df,val) : 
        labels = df.columns.tolist()

        # create dictionaries for each column as items of a list
        bxp_stats = df.apply(lambda x: {'med':x.med, 'q1':x.q1, 'q3':x.q3, 'whislo':x['min'], 'whishi':x['max']}, axis=0).tolist()

        # add the column names as labels to each dictionary entry
        for index, item in enumerate(bxp_stats):
            item.update({'label':labels[index]})

        fig, ax = plt.subplots()
        plt.figure(figsize=(15,3))
        ax.set_title(labels[0][0:-2])
        ax.bxp(bxp_stats, showfliers=False, vert=False,)
        # KO plt.axhline(x=val,color='red')
        # plt.show()
        barplot_chart = st.write(fig)
     
    # IMPORT MODEL
    import joblib
    with open('P7/modem_score_lg.pkl', 'rb') as f:
        model=joblib.load(f)
    #st.markdown(model, unsafe_allow_html = True)    
     
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    
    # IMPORT DES STATS
    dfStat=pd.read_csv('P7/df_BarplotInfo.csv', sep=';')
    dfStat.index = ['max','min','q1','q3','med']
    
    # LIGNE AU HASARD 
    lign=randint(0, dfExempleSize)   
    SK_ID_CURR=dfExemple.loc[lign,'SK_ID_CURR']
    dflign=dfExemple[dfExemple['SK_ID_CURR']==SK_ID_CURR]
    
    st.markdown(str(model)+' >> '+str(dfExempleSize)+' ['+str(lign)+']', unsafe_allow_html = True)  
    
    # --------------------------------------------- FORMULAIRE
    # 'SK_ID_CURR', 
    # 'PAYMENT_RATE', 
    # 'EXT_SOURCE_1', 
    # 'EXT_SOURCE_3', 
    # 'EXT_SOURCE_2', 
    # 'DAYS_BIRTH', 
    # 'AMT_ANNUITY', 
    # 'DAYS_EMPLOYED', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 
    # 'ACTIVE_DAYS_CREDIT_MAX', 
    # 'APPROVED_CNT_PAYMENT_MEAN', 
    # 'PREV_CNT_PAYMENT_MEAN', 
    # 'INSTAL_DPD_MEAN', 
    # 'DAYS_ID_PUBLISH', 
    # 'INSTAL_AMT_PAYMENT_MIN', 
    # 'ANNUITY_INCOME_PERC', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 
    # 'DAYS_EMPLOYED_PERC', 
    # 'AMT_CREDIT', 
    # 'POS_MONTHS_BALANCE_SIZE', 
    # 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 
    # 'AMT_GOODS_PRICE', 
    # 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 
    # 'INSTAL_AMT_PAYMENT_SUM', 
    # 'OWN_CAR_AGE', 
    # 'DAYS_REGISTRATION'

    listvardays=('DAYS_BIRTH', 'DAYS_EMPLOYED', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'ACTIVE_DAYS_CREDIT_MAX', 'DAYS_ID_PUBLISH', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'DAYS_EMPLOYED_PERC' ,'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'DAYS_REGISTRATION')
    
    i=0
    listVar=[]
    initial_date = "12/1/2022"
    initial_date2 = convert("2022-12-01")
    f='%Y-%m-%d'   
    
    with st.form("Predict_form"):
        for c in dflign :
            var=c
            listVar.append(c)
            val = dflign.iloc[0,i]
            if c != 'SK_ID_CURR' :
                if c in listvardays :
                    req_date = pd.to_datetime(initial_date) + pd.DateOffset(days=round(val))
                    globals()[var] = st.date_input(c, req_date)
                else :
                    tmpminmax = dfStat[[var+'_0',var+'_1']].filter(items = ['min','max'], axis=0).stack()
                    mini=tmpminmax.min()
                    maxi=tmpminmax.max()
                    # globals()[var] = st.number_input(c, val, min_value=mini ,max_value=tmpminmax.maxi)
                    # globals()[var] =st.slider(c, min_value=mini, max_value=maxi, value=val, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
                    # globals()[var] = st.slider(c, min_value=mini, max_value=maxi, value=val)
                    # st.markdown(mini, unsafe_allow_html = True)
                    # st.markdown(maxi, unsafe_allow_html = True)
                    txtTitr=c+' ['+str(mini)+' ; '+str(maxi)+']'
                    globals()[var] = st.number_input(txtTitr, val)
                    # globals()[var] = st.slider(txtTitr, min_value=mini, max_value=maxi, value=val)
            else : 
                globals()[var] = val
                
            i+=1
            
        submitted = st.form_submit_button("Submit")    
    
        # --------------------------------------------- PREDICTION
        # if st.button("Predict"):
        #     result = model.predict(dflign)
        #     st.success('The output is {}'.format(result))
        if submitted: 
            # st.markdown(listVar , unsafe_allow_html = True)
            tabinfo=[]
            for c in listVar :
                # st.markdown(c, unsafe_allow_html = True)
                # st.markdown(globals()[c], unsafe_allow_html = True) 
                if c in listvardays :
                    # st.markdown('DATE', unsafe_allow_html = True)
                    lifetime= convert(str(globals()[c])) -initial_date2
                    # st.markdown(lifetime.days , unsafe_allow_html = True)
                    tabinfo.append(lifetime.days)
                else :
                    tabinfo.append(globals()[c])
            # st.markdown(tabinfo, unsafe_allow_html = True) 
            info = pd.DataFrame([tabinfo], columns = listVar)
            # st.markdown(info.shape, unsafe_allow_html = True) 
            # st.markdown(len(tabinfo), unsafe_allow_html = True) 
            # st.markdown(len(listVar), unsafe_allow_html = True) 
            result = model.predict(info)
            st.success('The output is {}'.format(result))
            # st.markdown(dflign.shape, unsafe_allow_html = True) 
        
            # GRAPHIQUES
            list = dfStat.columns.tolist()[::2]
            for v in list :
                if v != 'OWN_CAR_AGE' :
                    v2=v[0:-1]+'1'
                    varval=v[0:-2]
                    print (v,' & ',v2)    
                    boxplotNyx2 (dfStat[[v,v2]],globals()[v[0:-2]])

def nyx_demo2():
    
     ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit NYX 0001 </h1>
    </div>    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)  

    
    from datetime import datetime
    def convert(date_time) :
        f='%Y-%m-%d'
        datetime_str = datetime.strptime(date_time,f)
        return datetime_str
    
    import matplotlib.pyplot as plt
    def boxplotNyx2 (df,val) : 
        labels = df.columns.tolist()

        # create dictionaries for each column as items of a list
        bxp_stats = df.apply(lambda x: {'med':x.med, 'q1':x.q1, 'q3':x.q3, 'whislo':x['min'], 'whishi':x['max']}, axis=0).tolist()

        # add the column names as labels to each dictionary entry
        for index, item in enumerate(bxp_stats):
            item.update({'label':labels[index]})

        fig, ax = plt.subplots()
        plt.figure(figsize=(15,3))
        ax.set_title(labels[0][0:-2])
        ax.bxp(bxp_stats, showfliers=False, vert=False,)
        # KO plt.axhline(x=val,color='red')
        # plt.show()
        barplot_chart = st.write(fig)
     
    # IMPORT MODEL
    import joblib
    with open('P7/modem_score_lg.pkl', 'rb') as f:
        model=joblib.load(f)
    #st.markdown(model, unsafe_allow_html = True)    
     
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    
    # IMPORT DES STATS
    dfStat=pd.read_csv('P7/df_BarplotInfo.csv', sep=';')
    dfStat.index = ['max','min','q1','q3','med']
    
    # LIGNE AU HASARD 
    lign=randint(0, dfExempleSize)   
    SK_ID_CURR=dfExemple.loc[lign,'SK_ID_CURR']
    dflign=dfExemple[dfExemple['SK_ID_CURR']==SK_ID_CURR]
    
    st.markdown(str(model)+' >> '+str(dfExempleSize)+' ['+str(lign)+']', unsafe_allow_html = True)  
    
    # --------------------------------------------- FORMULAIRE
    # 'SK_ID_CURR', 
    # 'PAYMENT_RATE', 
    # 'EXT_SOURCE_1', 
    # 'EXT_SOURCE_3', 
    # 'EXT_SOURCE_2', 
    # 'DAYS_BIRTH', 
    # 'AMT_ANNUITY', 
    # 'DAYS_EMPLOYED', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 
    # 'ACTIVE_DAYS_CREDIT_MAX', 
    # 'APPROVED_CNT_PAYMENT_MEAN', 
    # 'PREV_CNT_PAYMENT_MEAN', 
    # 'INSTAL_DPD_MEAN', 
    # 'DAYS_ID_PUBLISH', 
    # 'INSTAL_AMT_PAYMENT_MIN', 
    # 'ANNUITY_INCOME_PERC', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 
    # 'DAYS_EMPLOYED_PERC', 
    # 'AMT_CREDIT', 
    # 'POS_MONTHS_BALANCE_SIZE', 
    # 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 
    # 'AMT_GOODS_PRICE', 
    # 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 
    # 'INSTAL_AMT_PAYMENT_SUM', 
    # 'OWN_CAR_AGE', 
    # 'DAYS_REGISTRATION'

    listvardays=('DAYS_BIRTH', 'DAYS_EMPLOYED', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'ACTIVE_DAYS_CREDIT_MAX', 'DAYS_ID_PUBLISH', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'DAYS_EMPLOYED_PERC' ,'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'DAYS_REGISTRATION')
    
    i=0
    listVar=[]
    initial_date = "12/1/2022"
    initial_date2 = convert("2022-12-01")
    f='%Y-%m-%d'   
    dfConvert = pd.DataFrame(columns=['coef','decal'])    
    with st.form("Predict_form"):
        for c in dflign :
            var=c
            listVar.append(c)
            val = dflign.iloc[0,i]
            if c != 'SK_ID_CURR' :
                if c in listvardays :
                    req_date = pd.to_datetime(initial_date) + pd.DateOffset(days=round(val))
                    globals()[var] = st.date_input(c, req_date)
                else :
                    tmpminmax = dfStat[[var+'_0',var+'_1']].filter(items = ['min','max'], axis=0).stack()
                    mini=tmpminmax.min()
                    maxi=tmpminmax.max()
                    coef=(maxi-mini)/100
                    dfConvert.loc[i] = [coef,mini]
                    valConvert=round((val-mini)/coef)
                    txtTitr=c+' ['+str(mini)+' ; '+str(maxi)+'] '+str(valConvert)+' -- '+str(coef)+' -- '+str(val)
                    # st.markdown(txtTitr, unsafe_allow_html = True)
                    globals()[var] = st.slider(txtTitr, min_value=0, max_value=100, value=valConvert)
            else : 
                globals()[var] = val
                
            i+=1
            
        submitted = st.form_submit_button("Submit")    
    
        # --------------------------------------------- PREDICTION
        # if st.button("Predict"):
        #     result = model.predict(dflign)
        #     st.success('The output is {}'.format(result))
        i=0
        if submitted: 
            # st.markdown(listVar , unsafe_allow_html = True)
            tabinfo=[]
            for c in listVar :
                # st.markdown(c, unsafe_allow_html = True)
                # st.markdown(globals()[c], unsafe_allow_html = True) 
                if c in listvardays :
                    # st.markdown('DATE', unsafe_allow_html = True)
                    lifetime= convert(str(globals()[c])) -initial_date2
                    # st.markdown(lifetime.days , unsafe_allow_html = True)
                    tabinfo.append(lifetime.days)
                else :
                    valConvert=globals()[c]
                    valForm=dfConvert.iloc[i, 1]+(valConvert*dfConvert.iloc[i, 0])
                    tabinfo.append(globals()[c])
            i+=1
            # st.markdown(tabinfo, unsafe_allow_html = True) 
            info = pd.DataFrame([tabinfo], columns = listVar)
            # st.markdown(info.shape, unsafe_allow_html = True) 
            # st.markdown(len(tabinfo), unsafe_allow_html = True) 
            # st.markdown(len(listVar), unsafe_allow_html = True) 
            result = model.predict(info)
            st.success('The output is {}'.format(result))
            # st.markdown(dflign.shape, unsafe_allow_html = True) 
        
            # GRAPHIQUES
            list = dfStat.columns.tolist()[::2]
            for v in list :
                if v != 'OWN_CAR_AGE' :
                    v2=v[0:-1]+'1'
                    varval=v[0:-2]
                    print (v,' & ',v2)    
                    boxplotNyx2 (dfStat[[v,v2]],globals()[v[0:-2]])
                    
                    
                    
def nyx_df():
    
     ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit NYX 0001 Data exemples</h1>
    </div>    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)   

    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    st.markdown(dfExempleSize, unsafe_allow_html = True)   
    
    st.dataframe(dfExemple, use_container_width=st.session_state.use_container_width)
    


    
def nyx_df2():
    
     ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit NYX 0001 Data exemples</h1>
    </div>    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)  
    
    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    
    
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    st.markdown(dfExempleSize, unsafe_allow_html = True)   
    
    st.dataframe(dfExemple, use_container_width=st.session_state.use_container_width)
    
    
    # from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
    import AgGrid
    
    data= pd.read_csv('P7/X4Sample.csv', index_col=0) 
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        theme='blue', #Add theme color to the table
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        reload_data=True
    )

    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df    
    
def nyx_stat():
    
    ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit NYX 0001 Stat</h1>
    </div>    
    """
 
    
    import matplotlib.pyplot as plt
    def boxplotNyx (df) : 
        labels = df.columns.tolist()

        # create dictionaries for each column as items of a list
        bxp_stats = df.apply(lambda x: {'med':x.med, 'q1':x.q1, 'q3':x.q3, 'whislo':x['min'], 'whishi':x['max']}, axis=0).tolist()

        # add the column names as labels to each dictionary entry
        for index, item in enumerate(bxp_stats):
            item.update({'label':labels[index]})


        fig, ax = plt.subplots()
        plt.figure(figsize=(15,3))
        ax.set_title(labels[0][0:-2])
        ax.bxp(bxp_stats, showfliers=False, vert=False,);
        # plt.show()
        barplot_chart = st.write(fig)
    
    st.markdown(html_temp, unsafe_allow_html = True)  
    
    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    
    dfStat=pd.read_csv('P7/df_BarplotInfo.csv', sep=';')
    dfStat.index = ['max','min','q1','q3','med']
    st.dataframe(dfStat, use_container_width=st.session_state.use_container_width)
    
    list = dfStat.columns.tolist()[::2]
    for v in list :
        if v != 'OWN_CAR_AGE_0' :
            v2=v[0:-1]+'1'
            print (v,' & ',v2)    
            boxplotNyx (dfStat[[v,v2]])
        
def nyx_test():
    
    ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit NYX 0001 TEST FORMULAIRE</h1>
    </div>    
    """
    
    with st.form("my_form"):
        st.write("Inside the form")
        slider_val = st.slider("Form slider")
        checkbox_val = st.checkbox("Form checkbox")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("slider", slider_val, "checkbox", checkbox_val)

    st.write("Outside the form")
    
page_names_to_funcs = {
    "—": intro,
#   "Plotting Demo": plotting_demo,
#    "Mapping Demo": mapping_demo,
#    "DataFrame Demo": data_frame_demo,
    "NYX Demo": nyx_demo,
    "NYX Demo2": nyx_demo2,
    "NYX Stat": nyx_stat,
    "NYX Df": nyx_df,
    "NYX Df2": nyx_df2,
    "NYX TEST": nyx_test
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()