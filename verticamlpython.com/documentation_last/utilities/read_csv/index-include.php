<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="read_csv">read_csv<a class="anchor-link" href="#read_csv">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_csv</span><span class="p">(</span><span class="n">path</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
         <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
         <span class="n">schema</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;public&#39;</span><span class="p">,</span> 
         <span class="n">table_name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span> 
         <span class="n">sep</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;,&#39;</span><span class="p">,</span> 
         <span class="n">header</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
         <span class="n">header_names</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
         <span class="n">na_rep</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span> 
         <span class="n">quotechar</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span> 
         <span class="n">escape</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="se">\\</span><span class="s1">&#39;</span><span class="p">,</span> 
         <span class="n">genSQL</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
         <span class="n">parse_n_lines</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span>
         <span class="n">insert</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Ingests a CSV file using flex tables.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">path</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Absolute path where the CSV file is located.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">schema</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Schema where the CSV file will be ingested.</td> </tr>
    <tr> <td><div class="param_name">table_name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Final relation name.</td> </tr>
    <tr> <td><div class="param_name">sep</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Column separator.</td> </tr>
    <tr> <td><div class="param_name">header</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to False, the parameter 'header_names' will be used to name the different columns.</td> </tr>
    <tr> <td><div class="param_name">header_names</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the columns names.</td> </tr>
    <tr> <td><div class="param_name">na_rep</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Missing values representation.</td> </tr>
    <tr> <td><div class="param_name">quotechar</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Char which is enclosing the str values.</td> </tr>
    <tr> <td><div class="param_name">escape</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Separator between each record.</td> </tr>
    <tr> <td><div class="param_name">genSQL</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, the SQL code used to create the final table will be generated but not executed. It is a good way to change the final relation types or to customize the data ingestion.</td> </tr>
    <tr> <td><div class="param_name">parse_n_lines</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>If this parameter is greater than 0. A new file of 'parse_n_lines' lines will be created and ingested first to identify the data types. It will be then dropped and the entire file will be ingested. The data types identification will be less precise but this parameter can make the process faster if the file is heavy.</td> </tr>
<tr> <td><div class="param_name">insert</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, the data will be ingested to the input relation. Be sure that your file has a header corresponding to the name of the relation columns otherwise the ingestion will not work.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : The vDataFrame of the relation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># Gen the SQL needed to create the Table</span>
<span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span> 
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">quotechar</span> <span class="o">=</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
         <span class="n">na_rep</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
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
<pre>CREATE TABLE &#34;public&#34;.&#34;titanic_dataset&#34;(&#34;pclass&#34; Integer, &#34;survived&#34; Integer, &#34;name&#34; Varchar(164), &#34;sex&#34; Varchar(20), &#34;age&#34; Numeric(6,3), &#34;sibsp&#34; Integer, &#34;parch&#34; Integer, &#34;ticket&#34; Varchar(36), &#34;fare&#34; Numeric(10,5), &#34;cabin&#34; Varchar(30), &#34;embarked&#34; Varchar(20), &#34;boat&#34; Varchar(100), &#34;body&#34; Integer, &#34;home.dest&#34; Varchar(100));
COPY &#34;public&#34;.&#34;titanic_dataset&#34;(&#34;pclass&#34;, &#34;survived&#34;, &#34;name&#34;, &#34;sex&#34;, &#34;age&#34;, &#34;sibsp&#34;, &#34;parch&#34;, &#34;ticket&#34;, &#34;fare&#34;, &#34;cabin&#34;, &#34;embarked&#34;, &#34;boat&#34;, &#34;body&#34;, &#34;home.dest&#34;) FROM {} DELIMITER &#39;,&#39; NULL &#39;&#39; ENCLOSED BY &#39;&#34;&#39; ESCAPE AS &#39;\&#39; SKIP 1;
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Gen the SQL needed to create the Table &amp; Parses only 100 lines </span>
<span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span> 
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">quotechar</span> <span class="o">=</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
         <span class="n">na_rep</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
         <span class="n">parse_n_lines</span> <span class="o">=</span> <span class="mi">100</span><span class="p">,</span>
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
<pre>CREATE TABLE &#34;public&#34;.&#34;titanic_dataset&#34;(&#34;pclass&#34; Integer, &#34;survived&#34; Integer, &#34;name&#34; Varchar(130), &#34;sex&#34; Varchar(20), &#34;age&#34; Numeric(5,3), &#34;sibsp&#34; Integer, &#34;parch&#34; Integer, &#34;ticket&#34; Varchar(22), &#34;fare&#34; Numeric(10,5), &#34;cabin&#34; Varchar(22), &#34;embarked&#34; Varchar(20), &#34;boat&#34; Varchar(100), &#34;body&#34; Integer, &#34;home.dest&#34; Varchar(92));
COPY &#34;public&#34;.&#34;titanic_dataset&#34;(&#34;pclass&#34;, &#34;survived&#34;, &#34;name&#34;, &#34;sex&#34;, &#34;age&#34;, &#34;sibsp&#34;, &#34;parch&#34;, &#34;ticket&#34;, &#34;fare&#34;, &#34;cabin&#34;, &#34;embarked&#34;, &#34;boat&#34;, &#34;body&#34;, &#34;home.dest&#34;) FROM {} DELIMITER &#39;,&#39; NULL &#39;&#39; ENCLOSED BY &#39;&#34;&#39; ESCAPE AS &#39;\&#39; SKIP 1;
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Ingests the CSV file</span>
<span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span> 
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">quotechar</span> <span class="o">=</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
         <span class="n">na_rep</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
         <span class="n">parse_n_lines</span> <span class="o">=</span> <span class="mi">100</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic_dataset&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">B5</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">St Louis, MO</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">29.000</td><td style="border: 1px solid white;">Allen, Miss. Elisabeth Walton</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">211.33750</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">24160</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">11</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0.920</td><td style="border: 1px solid white;">Allison, Master. Hudson Trevor</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_dataset, Number of rows: 1234, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Inserts a new file in an existing table</span>
<span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span> 
         <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
         <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
         <span class="n">quotechar</span> <span class="o">=</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span>
         <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
         <span class="n">na_rep</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
         <span class="n">parse_n_lines</span> <span class="o">=</span> <span class="mi">100</span><span class="p">,</span>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">211.33750</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">24160</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">B5</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">St Louis, MO</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">29.000</td><td style="border: 1px solid white;">Allen, Miss. Elisabeth Walton</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">11</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0.920</td><td style="border: 1px solid white;">Allison, Master. Hudson Trevor</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">1</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_dataset, Number of rows: 2468, Number of columns: 14</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="See-Also">See Also<a class="anchor-link" href="#See-Also">&#182;</a></h3><table id="seealso">
    <tr><td><a href="../read_json/index.php">read_json</a></td> <td>Ingests a JSON file in the Vertica DB.</td></tr>
</table>
</div>
</div>
</div>