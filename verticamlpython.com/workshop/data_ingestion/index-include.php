<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Data-Ingestion">Data Ingestion<a class="anchor-link" href="#Data-Ingestion">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The First Step of the Data Science Process (Excluding Business Understanding) is the Data Ingestion. To do Data Science, we need data and it is important to be able to ingest different types of formats. Vertica allows the ingestion of many data files thanks to different built-in parsers.</p>
<p>Most of the time we know in advance the file types and we need to write the entire SQL query to ingest it. However, sometimes we don't know the columns names and types in advance. To solve this problem, Vertica allows the users to create Flex Tables. They are efficient ways to ingest any data file without knowing in advance its columns types or even its structure.</p>
<p>Vertica ML Python is using Flex Tables to allow the auto-ingestion of JSON and CSV files. For the other files types, it is advise to use direct SQL queries to ingest them. Becareful when using the following functions as the data types detected may not be optimal and it is always preferable to write SQL queries using optimized types and segmentations.</p>
<p>It is important to remember that Vertica ML Python is using Vertica SQL in back-end so by optimizing table structure you are increasing Vertica ML Python performance.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Ingesting-CSV">Ingesting CSV<a class="anchor-link" href="#Ingesting-CSV">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>CSV is the favourite data scientists format. It has an internal structure which makes it easy to ingest. To ingest a CSV file, we will use the 'read_csv' function.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">read_csv</span>
<span class="n">help</span><span class="p">(</span><span class="n">read_csv</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function read_csv in module vertica_ml_python.utilities:

read_csv(path:str, cursor=None, schema:str=&#39;public&#39;, table_name:str=&#39;&#39;, sep:str=&#39;,&#39;, header:bool=True, header_names:list=[], na_rep:str=&#39;&#39;, quotechar:str=&#39;&#34;&#39;, escape:str=&#39;\\&#39;, genSQL:bool=False, parse_n_lines:int=-1, insert:bool=False)
    ---------------------------------------------------------------------------
    Ingests a CSV file using flex tables.
    
    Parameters
    ----------
    path: str
            Absolute path where the CSV file is located.
    cursor: DBcursor, optional
            Vertica DB cursor.
    schema: str, optional
            Schema where the CSV file will be ingested.
    table_name: str, optional
            Final relation name.
    sep: str, optional
            Column separator.
    header: bool, optional
            If set to False, the parameter &#39;header_names&#39; will be used to name the 
            different columns.
    header_names: list, optional
            List of the columns names.
    na_rep: str, optional
            Missing values representation.
    quotechar: str, optional
            Char which is enclosing the str values.
    escape: str, optional
            Separator between each record.
    genSQL: bool, optional
            If set to True, the SQL code used to create the final table will be 
            generated but not executed. It is a good way to change the final
            relation types or to customize the data ingestion.
    parse_n_lines: int, optional
            If this parameter is greater than 0. A new file of &#39;parse_n_lines&#39; lines
            will be created and ingested first to identify the data types. It will be
            then dropped and the entire file will be ingested. The data types identification
            will be less precise but this parameter can make the process faster if the
            file is heavy.
    insert: bool, optional
            If set to True, the data will be ingested to the input relation. Be sure
            that your file has a header corresponding to the name of the relation
            columns otherwise the ingestion will not work.
    
    Returns
    -------
    vDataFrame
            The vDataFrame of the relation.
    
    See Also
    --------
    read_json : Ingests a JSON file in the Vertica DB.

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You can easily ingest a CSV file by entering the correct parameters.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic&quot;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">24160</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">St Louis, MO</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">211.3375</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">Allen, Miss. Elisabeth Walton</td><td style="border: 1px solid white;">B5</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">11</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.92</td><td style="border: 1px solid white;">Allison, Master. Hudson Trevor</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[16]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>If no schema is indicated as parameter, the 'public' schema will be used. 
If 'table_name' is not defined, the name of the final relation will correspond to the name of the CSV file.</p>
<p>It is also possible to not ingest the file and only to generate the SQL query which can be used to create the final relation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic&quot;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
         <span class="n">genSQL</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>CREATE TABLE &#34;public&#34;.&#34;titanic&#34;(&#34;pclass&#34; Integer, &#34;survived&#34; Integer, &#34;name&#34; Varchar(164), &#34;sex&#34; Varchar(20), &#34;age&#34; Numeric(6,3), &#34;sibsp&#34; Integer, &#34;parch&#34; Integer, &#34;ticket&#34; Varchar(36), &#34;fare&#34; Numeric(10,5), &#34;cabin&#34; Varchar(30), &#34;embarked&#34; Varchar(20), &#34;boat&#34; Varchar(100), &#34;body&#34; Integer, &#34;home.dest&#34; Varchar(100));
COPY &#34;public&#34;.&#34;titanic&#34;(&#34;pclass&#34;, &#34;survived&#34;, &#34;name&#34;, &#34;sex&#34;, &#34;age&#34;, &#34;sibsp&#34;, &#34;parch&#34;, &#34;ticket&#34;, &#34;fare&#34;, &#34;cabin&#34;, &#34;embarked&#34;, &#34;boat&#34;, &#34;body&#34;, &#34;home.dest&#34;) FROM {} DELIMITER &#39;,&#39; NULL &#39;&#39; ENCLOSED BY &#39;&#34;&#39; ESCAPE AS &#39;\&#39; SKIP 1;
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You can also use the parameter 'insert' to insert new data in the existing relation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic&quot;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
         <span class="n">insert</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">24160</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">St Louis, MO</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">211.3375</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">Allen, Miss. Elisabeth Walton</td><td style="border: 1px solid white;">B5</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">11</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.92</td><td style="border: 1px solid white;">Allison, Master. Hudson Trevor</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[22]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 2468, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Ingesting-JSON">Ingesting JSON<a class="anchor-link" href="#Ingesting-JSON">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>JSON is also a popular format and you can ingest JSON files using the 'read_json' function.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">read_csv</span>
<span class="n">help</span><span class="p">(</span><span class="n">read_json</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function read_json in module vertica_ml_python.utilities:

read_json(path:str, cursor=None, schema:str=&#39;public&#39;, table_name:str=&#39;&#39;, usecols:list=[], new_name:dict={}, insert:bool=False)
    ---------------------------------------------------------------------------
    Ingests a JSON file using flex tables.
    
    Parameters
    ----------
    path: str
            Absolute path where the JSON file is located.
    cursor: DBcursor, optional
            Vertica DB cursor.
    schema: str, optional
            Schema where the JSON file will be ingested.
    table_name: str, optional
            Final relation name.
    usecols: list, optional
            List of the JSON parameters to ingest. The other ones will be ignored. If
            empty all the JSON parameters will be ingested.
    new_name: dict, optional
            Dictionary of the new columns name. If the JSON file is nested, it is advised
            to change the final names as special characters will be included.
            For example, {&#34;param&#34;: {&#34;age&#34;: 3, &#34;name&#34;: Badr}, &#34;date&#34;: 1993-03-11} will 
            create 3 columns: &#34;param.age&#34;, &#34;param.name&#34; and &#34;date&#34;. You can rename these 
            columns using the &#39;new_name&#39; parameter with the following dictionary:
            {&#34;param.age&#34;: &#34;age&#34;, &#34;param.name&#34;: &#34;name&#34;}
    insert: bool, optional
            If set to True, the data will be ingested to the input relation. The JSON
            parameters must be the same than the input relation otherwise they will
            not be ingested.
    
    Returns
    -------
    vDataFrame
            The vDataFrame of the relation.
    
    See Also
    --------
    read_csv : Ingests a CSV file in the Vertica DB.

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This function will work the same way as 'read_csv' but it has less parameter due to the standardization of the JSON format.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_json</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">,</span>
          <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
          <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>record_timestamp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.passengerid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>recordid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>datasetid</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Collander, Mr. Erik Gustaf</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">343</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">13.0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">248740</td><td style="border: 1px solid white;">835634b93c8f759537a89daa01c3c3658e934617</td><td style="border: 1px solid white;">28.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Moen, Mr. Sigurd Hansen</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">76</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7.65</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">348123</td><td style="border: 1px solid white;">97941a419e5cf6a4bb65147a7a21d7025c8a6e1b</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">F G73</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Jensen, Mr. Hans Peder</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">641</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7.8542</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">350050</td><td style="border: 1px solid white;">b762da1fa9f7f7765bc14006d9f5b8fc1d2d5177</td><td style="border: 1px solid white;">20.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Palsson, Mrs. Nils (Alma Cornelia Berglund)</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">568</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">21.075</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">349909</td><td style="border: 1px solid white;">dc455b086d203605705820911c0aaa98467bcd41</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Davidson, Mr. Thornton</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">672</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">52.0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">F.C. 12750</td><td style="border: 1px solid white;">5aa00b39a93376656528f1c7d929a297e31e1a20</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;">B71</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 891, Number of columns: 15</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Our 'JSON' file was nested which leaded to the creation of columns names having a dot ('.') separator. You can use the parameters 'usecols' and 'new_name' to only select the needed columns and rename them before the ingestion.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_json</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">,</span>
          <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
          <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic&quot;</span><span class="p">,</span>
          <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;fields.survived&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;fields.pclass&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;fields.fare&quot;</span><span class="p">],</span>
          <span class="n">new_name</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;fields.survived&quot;</span><span class="p">:</span> <span class="s2">&quot;survived&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;fields.pclass&quot;</span><span class="p">:</span> <span class="s2">&quot;pclass&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;fields.fare&quot;</span><span class="p">:</span> <span class="s2">&quot;fare&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">13.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">False</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">7.65</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">False</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">7.8542</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">False</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">21.075</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">False</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">52.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">False</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[31]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 891, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You are now ready to understand the Vertica ML Python Data Exploration functionalities.</p>

</div>
</div>
</div>