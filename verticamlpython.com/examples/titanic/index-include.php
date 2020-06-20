<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Titanic">Titanic<a class="anchor-link" href="#Titanic">&#182;</a></h1><p>This notebook is an example on how to work with the Vertica ML Python Module. We will use the Titanic dataset to introduce the library. The purpose is to predict the passengers survival. You can download the Jupyter Notebook of the study by clicking <a href="titanic.ipynb">here</a>.</p>
<h2 id="Initialization">Initialization<a class="anchor-link" href="#Initialization">&#182;</a></h2><p>To avoid redundant cursors creation, let's create an auto connection.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.connections.connect</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">new_auto_connection</span><span class="p">({</span><span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;port&quot;</span><span class="p">:</span> <span class="s2">&quot;5433&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;database&quot;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;password&quot;</span><span class="p">:</span> <span class="s2">&quot;PPpdmzLX&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;user&quot;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">},</span> 
                    <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;vertica_python&quot;</span><span class="p">,</span> 
                    <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;VML&quot;</span><span class="p">)</span>
<span class="n">change_auto_connection</span><span class="p">(</span><span class="s2">&quot;VML&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Use the following command to allow Matplotlib to display graphic.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">%</span><span class="k">matplotlib</span> inline
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's load the Titanic dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_titanic</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">load_titanic</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">titanic</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 14
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
<h2 id="Data-Exploration-and-Preparation">Data Exploration and Preparation<a class="anchor-link" href="#Data-Exploration-and-Preparation">&#182;</a></h2><p>Let's explore the data by displaying descriptive statistics of all the columns.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;categorical&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>dtype</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>top</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>top_percent</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">63.533</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">varchar(100)</td><td style="border: 1px solid white;">26</td><td style="border: 1px solid white;">439</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">64.425</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"ticket"</b></td><td style="border: 1px solid white;">varchar(36)</td><td style="border: 1px solid white;">887</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">CA. 2343</td><td style="border: 1px solid white;">0.81</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"embarked"</b></td><td style="border: 1px solid white;">varchar(20)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1232</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">70.746</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"home.dest"</b></td><td style="border: 1px solid white;">varchar(100)</td><td style="border: 1px solid white;">359</td><td style="border: 1px solid white;">706</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">42.788</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">7</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">67.747</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">numeric(10,5)</td><td style="border: 1px solid white;">277</td><td style="border: 1px solid white;">1233</td><td style="border: 1px solid white;">8.05000</td><td style="border: 1px solid white;">4.7</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">varchar(20)</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">65.964</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"body"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">118</td><td style="border: 1px solid white;">118</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">90.438</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">53.728</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="border: 1px solid white;">numeric(6,3)</td><td style="border: 1px solid white;">96</td><td style="border: 1px solid white;">997</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">19.206</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"name"</b></td><td style="border: 1px solid white;">varchar(164)</td><td style="border: 1px solid white;">1232</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">Connolly, Miss. Kate</td><td style="border: 1px solid white;">0.162</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"cabin"</b></td><td style="border: 1px solid white;">varchar(30)</td><td style="border: 1px solid white;">182</td><td style="border: 1px solid white;">286</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">76.823</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">1234</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">76.904</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The column "body" is useless (it is only the ID of the passengers).
The column "home.dest" will not influence the survival, it is just from where the passengers embarked and where they are going to. We can have the same conclusion with "embarked" which is the port of embarkation. The column 'ticket' which is the ticket ID will also not give us information on the survival.</p>
<p>Let's analyze the columns "name" and "cabin to see if we can extract some information. Let's first look at the passengers names.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.preprocessing</span> <span class="k">import</span> <span class="n">CountVectorizer</span>
<span class="n">CountVectorizer</span><span class="p">(</span><span class="s2">&quot;name_voc&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;titanic&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;Name&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>token</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>df</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cnt</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>rnk</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">mr</td><td style="border: 1px solid white;">0.148163100524828421</td><td style="border: 1px solid white;">734</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">miss</td><td style="border: 1px solid white;">0.046023415421881308</td><td style="border: 1px solid white;">228</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">mrs</td><td style="border: 1px solid white;">0.037343560758982640</td><td style="border: 1px solid white;">185</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">william</td><td style="border: 1px solid white;">0.016148566814695196</td><td style="border: 1px solid white;">80</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">john</td><td style="border: 1px solid white;">0.013726281792490916</td><td style="border: 1px solid white;">68</td><td style="border: 1px solid white;">5</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: name_voc, Number of rows: 1841, Number of columns: 4</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>From the "name" it is possible to extract the title of the passengers. Let's now look at the "cabins".</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.preprocessing</span> <span class="k">import</span> <span class="n">CountVectorizer</span>
<span class="n">CountVectorizer</span><span class="p">(</span><span class="s2">&quot;cabin_voc&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;titanic&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;cabin&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>token</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>df</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cnt</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>rnk</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0.733746130030959752</td><td style="border: 1px solid white;">948</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">f</td><td style="border: 1px solid white;">0.006191950464396285</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">c27</td><td style="border: 1px solid white;">0.004643962848297214</td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">c25</td><td style="border: 1px solid white;">0.004643962848297214</td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">c23</td><td style="border: 1px solid white;">0.004643962848297214</td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">3</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: cabin_voc, Number of rows: 199, Number of columns: 4</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can extract the cabin position (the letter which reprent the position in the boat) and look at the number of occurences.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">CountVectorizer</span><span class="p">(</span><span class="s2">&quot;cabin_voc&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;titanic&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;cabin&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span>
                <span class="p">)[</span><span class="s2">&quot;token&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">str_slice</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span>
                <span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;token&quot;</span><span class="p">],</span> <span class="n">expr</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;SUM(cnt)&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">30</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>token</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SUM</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">a</td><td style="border: 1px solid white;">20</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">c</td><td style="border: 1px solid white;">113</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">e</td><td style="border: 1px solid white;">43</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">g</td><td style="border: 1px solid white;">9</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">948</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">b</td><td style="border: 1px solid white;">92</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">d</td><td style="border: 1px solid white;">47</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">f</td><td style="border: 1px solid white;">19</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">t</td><td style="border: 1px solid white;">1</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 9, Number of columns: 2</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The NULL values possibly represent passengers having no cabin (MNAR = Missing values not at random). The same for the column "boat" where NULL values represent passengers who have a dedicated "lifeboat". We can drop the useless columns and encode the others.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s2">&quot;body&quot;</span><span class="p">,</span> <span class="s2">&quot;home.dest&quot;</span><span class="p">,</span> <span class="s2">&quot;embarked&quot;</span><span class="p">,</span> <span class="s2">&quot;ticket&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[8]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;cabin&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">str_slice</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)[</span><span class="s2">&quot;name&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">str_extract</span><span class="p">(</span>
        <span class="s1">&#39; ([A-Za-z]+)\.&#39;</span><span class="p">)[</span><span class="s2">&quot;boat&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span>
        <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;0ifnull&quot;</span><span class="p">)[</span><span class="s2">&quot;cabin&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="s2">&quot;No Cabin&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>795 element(s) was/were filled
948 element(s) was/were filled
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">A</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">No Cabin</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can notice that our assumptions about the cabin are wrong as the first class passengers should have a cabin. This column has missing values at random (MAR) and too much. Let's drop it.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;cabin&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">drop</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 9</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's look at descriptive statistics of the entire Virtual Dataframe.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">statistics</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>skewness</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>kurtosis</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>stddev</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>min</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>10%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>25%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>median</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>75%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>90%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>max</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">0.56300284427369</td><td style="border: 1px solid white;">-1.68576262213743</td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">0.364667747163695</td><td style="border: 1px solid white;">0.481532018641288</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">0.60334064583739</td><td style="border: 1px solid white;">-1.63863851358893</td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">0.355753646677472</td><td style="border: 1px solid white;">0.478935143777661</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">3.7597831472411</td><td style="border: 1px solid white;">19.2138853382802</td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">0.504051863857374</td><td style="border: 1px solid white;">1.04111727241629</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">8.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">4.30069918891405</td><td style="border: 1px solid white;">26.2543152552867</td><td style="border: 1px solid white;">1233.0</td><td style="border: 1px solid white;">33.9637936739659</td><td style="border: 1px solid white;">52.6460729831293</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">7.5892</td><td style="border: 1px solid white;">7.8958</td><td style="border: 1px solid white;">14.4542</td><td style="border: 1px solid white;">31.3875</td><td style="border: 1px solid white;">79.13</td><td style="border: 1px solid white;">512.3292</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">-0.576258567091907</td><td style="border: 1px solid white;">-1.34962169484619</td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">2.28444084278768</td><td style="border: 1px solid white;">0.842485636190292</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">3.0</td><td style="border: 1px solid white;">3.0</td><td style="border: 1px solid white;">3.0</td><td style="border: 1px solid white;">3.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="border: 1px solid white;">0.408876460779437</td><td style="border: 1px solid white;">0.15689691331997</td><td style="border: 1px solid white;">997.0</td><td style="border: 1px solid white;">30.1524573721163</td><td style="border: 1px solid white;">14.4353046299159</td><td style="border: 1px solid white;">0.33</td><td style="border: 1px solid white;">14.5</td><td style="border: 1px solid white;">21.0</td><td style="border: 1px solid white;">28.0</td><td style="border: 1px solid white;">39.0</td><td style="border: 1px solid white;">50.0</td><td style="border: 1px solid white;">80.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">3.79801928269975</td><td style="border: 1px solid white;">22.6438022640172</td><td style="border: 1px solid white;">1234.0</td><td style="border: 1px solid white;">0.378444084278768</td><td style="border: 1px solid white;">0.868604707790392</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">9.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[11]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This method will help us to understand a bit more our data. For example, we can notice that the "age" of the passengers follows more or less a normal distribution (kurtosis and skewness around 0). Let's draw the "age" histogram to verify our hypothesis.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can also do a Jarque Bera test to prove our hypothesis.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span><span class="o">.</span><span class="n">testjb</span><span class="p">(</span><span class="s2">&quot;age&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>We test the following hypothesis:
(H0) The distribution of &#34;age&#34; is not normal
(H1) The distribution of &#34;age&#34; is normal
👍 - The distribution of &#34;age&#34; might be normal
jb = 28.5338631758186
p_value = 0.0
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(True, 28.5338631758186, 0.0)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Many other relevant information can be noticed. The column 'fare' has many outliers (The maximum 512.33 is much greater than the 9th decile 79.13). Most of the passengers traveled in 3rd class (median of pclass = 3) and much more...</p>
<p>As 'sibsp' represents the number of siblings and parch the number of parents and children, it can be relevant to create a new feature: 'family_size'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">eval</span><span class="p">(</span><span class="s2">&quot;family_size&quot;</span><span class="p">,</span> <span class="s2">&quot;parch + sibsp + 1&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[14]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's deal with the outliers. There are many methods to find them (LocalOutlier Factor, DBSCAN, KMeans...) but we will just winsorize the 'fare' distribution which is subject to huge anomalies.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;fare&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fill_outliers</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;winsorize&quot;</span><span class="p">,</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.03</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.5500000000000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.5500000000000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.5500000000000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0E-13</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.5042000000000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[15]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's encode the column 'sex' to be able to use it with numerical methods.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">label_encode</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.5500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.5500000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.5500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">4</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0E-13</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.5042000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[16]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The column 'age' has too many missing values and as most of the ML algorithms do not handle them, we need to use imputation techniques. Let's fill the missing values using the average 'age' of the passengers having the same 'pclass' and the same 'sex'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;mean&quot;</span><span class="p">,</span> <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;pclass&quot;</span><span class="p">,</span> <span class="s2">&quot;sex&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>237 element(s) was/were filled
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">135.6333000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">36.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">134.5000000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">26.5500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">21.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">211.5000000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">50.0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">164.8667000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">45.0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's draw the correlation matrix to see the variables links.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;spearman&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sex"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>"family_size"</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.949194918072845</td><td style="border: 1px solid white;">0.0934303518392484</td><td style="border: 1px solid white;">0.322278285934243</td><td style="border: 1px solid white;">-0.528849820180154</td><td style="border: 1px solid white;">-0.335634606444129</td><td style="border: 1px solid white;">-0.00336824465604963</td><td style="border: 1px solid white;">0.171691455579124</td><td style="border: 1px solid white;">0.185599315104602</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">0.949194918072845</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.100652268255772</td><td style="border: 1px solid white;">0.334910192561693</td><td style="border: 1px solid white;">-0.484361066873315</td><td style="border: 1px solid white;">-0.344146454498079</td><td style="border: 1px solid white;">0.00421157947473071</td><td style="border: 1px solid white;">0.175393711669708</td><td style="border: 1px solid white;">0.190320484343677</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">0.0934303518392484</td><td style="border: 1px solid white;">0.100652268255772</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.444010325371998</td><td style="border: 1px solid white;">-0.184712709381236</td><td style="border: 1px solid white;">-0.0691851178284916</td><td style="border: 1px solid white;">-0.116480956583223</td><td style="border: 1px solid white;">0.436524985907755</td><td style="border: 1px solid white;">0.854550816071704</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">0.322278285934243</td><td style="border: 1px solid white;">0.334910192561693</td><td style="border: 1px solid white;">0.444010325371998</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">-0.264115871968389</td><td style="border: 1px solid white;">-0.722423468046863</td><td style="border: 1px solid white;">0.228529791943121</td><td style="border: 1px solid white;">0.402198981036797</td><td style="border: 1px solid white;">0.52665819103252</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">-0.528849820180154</td><td style="border: 1px solid white;">-0.484361066873315</td><td style="border: 1px solid white;">-0.184712709381236</td><td style="border: 1px solid white;">-0.264115871968389</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.150967389146603</td><td style="border: 1px solid white;">0.0659428426025525</td><td style="border: 1px solid white;">-0.246951353519755</td><td style="border: 1px solid white;">-0.284892754418329</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">-0.335634606444129</td><td style="border: 1px solid white;">-0.344146454498079</td><td style="border: 1px solid white;">-0.0691851178284916</td><td style="border: 1px solid white;">-0.722423468046863</td><td style="border: 1px solid white;">0.150967389146603</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">-0.447455588651876</td><td style="border: 1px solid white;">-0.0429882472107741</td><td style="border: 1px solid white;">-0.109035609183308</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="border: 1px solid white;">-0.00336824465604963</td><td style="border: 1px solid white;">0.00421157947473071</td><td style="border: 1px solid white;">-0.116480956583223</td><td style="border: 1px solid white;">0.228529791943121</td><td style="border: 1px solid white;">0.0659428426025525</td><td style="border: 1px solid white;">-0.447455588651876</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">-0.174615632035404</td><td style="border: 1px solid white;">-0.133316437156597</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">0.171691455579124</td><td style="border: 1px solid white;">0.175393711669708</td><td style="border: 1px solid white;">0.436524985907755</td><td style="border: 1px solid white;">0.402198981036797</td><td style="border: 1px solid white;">-0.246951353519755</td><td style="border: 1px solid white;">-0.0429882472107741</td><td style="border: 1px solid white;">-0.174615632035404</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.776055413048162</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"family_size"</b></td><td style="border: 1px solid white;">0.185599315104602</td><td style="border: 1px solid white;">0.190320484343677</td><td style="border: 1px solid white;">0.854550816071704</td><td style="border: 1px solid white;">0.52665819103252</td><td style="border: 1px solid white;">-0.284892754418329</td><td style="border: 1px solid white;">-0.109035609183308</td><td style="border: 1px solid white;">-0.133316437156597</td><td style="border: 1px solid white;">0.776055413048162</td><td style="border: 1px solid white;">1.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The fare is very correlated to the family size. It is normal as the bigger the family is, the greater the number of tickets they have to buy will be (so the 'fare' as well). The survival is very correlated to the 'boat' variable. To avoid predictions only based on one variable and to increase the generality, we must split the study into 2 use cases.</p>
<p><ul>
    <li>Passengers having no lifeboat</li>
    <li>Passengers having a lifeboat</li>
</ul>
We did a lot of operations to clean this data and nothing was saved in the DB ! We can look at the Virtual Dataframe relation to be sure.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">titanic</span><span class="o">.</span><span class="n">current_relation</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>(
   SELECT
     &#34;survived&#34;,
     &#34;boat&#34;,
     &#34;sibsp&#34;,
     &#34;fare&#34;,
     &#34;sex&#34;,
     &#34;pclass&#34;,
     COALESCE(&#34;age&#34;, AVG(&#34;age&#34;) OVER (PARTITION BY &#34;pclass&#34;, &#34;sex&#34;)) AS &#34;age&#34;,
     &#34;name&#34;,
     &#34;parch&#34;,
     &#34;family_size&#34; 
   FROM
 (
   SELECT
     &#34;survived&#34;,
     DECODE(&#34;boat&#34;, NULL, 0, 1) AS &#34;boat&#34;,
     &#34;sibsp&#34;,
     (CASE WHEN &#34;fare&#34; &lt; -176.6204982585513 THEN -176.6204982585513 WHEN &#34;fare&#34; &gt; 244.5480856064831 THEN 244.5480856064831 ELSE &#34;fare&#34; END) AS &#34;fare&#34;,
     DECODE(&#34;sex&#34;, &#39;female&#39;, 0, &#39;male&#39;, 1, 2) AS &#34;sex&#34;,
     &#34;pclass&#34;,
     &#34;age&#34;,
     REGEXP_SUBSTR(&#34;name&#34;, &#39; ([A-Za-z]+)\.&#39;) AS &#34;name&#34;,
     &#34;parch&#34;,
     parch + sibsp + 1 AS &#34;family_size&#34; 
   FROM
 (
   SELECT
     &#34;survived&#34;,
     &#34;boat&#34;,
     &#34;sibsp&#34;,
     &#34;fare&#34;,
     &#34;sex&#34;,
     &#34;pclass&#34;,
     &#34;age&#34;,
     &#34;name&#34;,
     &#34;parch&#34; 
   FROM
 &#34;public&#34;.&#34;titanic&#34;) 
VERTICA_ML_PYTHON_SUBTABLE) 
VERTICA_ML_PYTHON_SUBTABLE) 
VERTICA_ML_PYTHON_SUBTABLE
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
<p>Let see what's happening when we aggregate and turn on the SQL.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">sql_on_off</span><span class="p">()</span><span class="o">.</span><span class="n">avg</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<h4 style = 'color : #444444; text-decoration : underline;'>Computes the different aggregations.</h4>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
 &emsp;  SELECT <br> &emsp;  &emsp;  '0.364667747163695', <br> &emsp;  &emsp;  '0.355753646677472', <br> &emsp;  &emsp;  '0.504051863857374', <br> &emsp;  &emsp;  AVG("fare"), <br> &emsp;  &emsp;  AVG("sex"), <br> &emsp;  &emsp;  '2.28444084278768', <br> &emsp;  &emsp;  AVG("age"), <br> &emsp;  &emsp;  '0.378444084278768', <br> &emsp;  &emsp;  AVG("family_size") &emsp; <br> &emsp;  FROM <br> ( <br> &emsp;  SELECT <br> &emsp;  &emsp;  "survived", <br> &emsp;  &emsp;  "boat", <br> &emsp;  &emsp;  "sibsp", <br> &emsp;  &emsp;  "fare", <br> &emsp;  &emsp;  "sex", <br> &emsp;  &emsp;  "pclass", <br> &emsp;  &emsp;  COALESCE("age", AVG("age") OVER (PARTITION BY "pclass", "sex")) AS "age", <br> &emsp;  &emsp;  "name", <br> &emsp;  &emsp;  "parch", <br> &emsp;  &emsp;  "family_size" &emsp; <br> &emsp;  FROM <br> ( <br> &emsp;  SELECT <br> &emsp;  &emsp;  "survived", <br> &emsp;  &emsp;  DECODE("boat", NULL, 0, 1) AS "boat", <br> &emsp;  &emsp;  "sibsp", <br> &emsp;  &emsp;  (CASE WHEN "fare" < -176.6204982585513 THEN -176.6204982585513 WHEN "fare" > 244.5480856064831 THEN 244.5480856064831 ELSE "fare" END) AS "fare", <br> &emsp;  &emsp;  DECODE("sex", 'female', 0, 'male', 1, 2) AS "sex", <br> &emsp;  &emsp;  "pclass", <br> &emsp;  &emsp;  "age", <br> &emsp;  &emsp;  REGEXP_SUBSTR("name", ' ([A-Za-z]+)\.') AS "name", <br> &emsp;  &emsp;  "parch", <br> &emsp;  &emsp;  parch + sibsp + 1 AS "family_size" &emsp; <br> &emsp;  FROM <br> ( <br> &emsp;  SELECT <br> &emsp;  &emsp;  "survived", <br> &emsp;  &emsp;  "boat", <br> &emsp;  &emsp;  "sibsp", <br> &emsp;  &emsp;  "fare", <br> &emsp;  &emsp;  "sex", <br> &emsp;  &emsp;  "pclass", <br> &emsp;  &emsp;  "age", <br> &emsp;  &emsp;  "name", <br> &emsp;  &emsp;  "parch" &emsp; <br> &emsp;  FROM <br> "public"."titanic") &emsp; <br>VERTICA_ML_PYTHON_SUBTABLE) &emsp; <br>VERTICA_ML_PYTHON_SUBTABLE) &emsp; <br>VERTICA_ML_PYTHON_SUBTABLE LIMIT 1
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<div style = 'border : 1px dashed black; width : 100%'></div>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"survived"</b></td><td style="border: 1px solid white;">0.364667747163695</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"boat"</b></td><td style="border: 1px solid white;">0.355753646677472</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sibsp"</b></td><td style="border: 1px solid white;">0.504051863857374</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"fare"</b></td><td style="border: 1px solid white;">32.9113074018842</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"sex"</b></td><td style="border: 1px solid white;">0.659643435980551</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"pclass"</b></td><td style="border: 1px solid white;">2.28444084278768</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"age"</b></td><td style="border: 1px solid white;">29.717623352014</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"parch"</b></td><td style="border: 1px solid white;">0.378444084278768</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"family_size"</b></td><td style="border: 1px solid white;">1.88249594813614</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[20]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Vertica ML Python will do SQL code generation during the entire process and it will keep in mind all the users modifications. Besides, it will store already computed aggregations to not compute them twice. The catalog will be updated in case of filtering or modifications which will affect the concern columns.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic</span><span class="o">.</span><span class="n">sql_on_off</span><span class="p">()</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The vDataFrame was modified many times: 
 * {Tue Apr 21 22:59:34 2020} [Drop]: vcolumn &#34;body&#34; was deleted from the vDataFrame.
 * {Tue Apr 21 22:59:34 2020} [Drop]: vcolumn &#34;home.dest&#34; was deleted from the vDataFrame.
 * {Tue Apr 21 22:59:34 2020} [Drop]: vcolumn &#34;embarked&#34; was deleted from the vDataFrame.
 * {Tue Apr 21 22:59:34 2020} [Drop]: vcolumn &#34;ticket&#34; was deleted from the vDataFrame.
 * {Tue Apr 21 22:59:34 2020} [SUBSTR(, 1, 1)]: The vcolumn &#39;cabin&#39; was transformed with the func &#39;x -&gt; SUBSTR(x, 1, 1)&#39;.
 * {Tue Apr 21 22:59:34 2020} [REGEXP_SUBSTR(, &#39; ([A-Za-z]+)\.&#39;)]: The vcolumn &#39;name&#39; was transformed with the func &#39;x -&gt; REGEXP_SUBSTR(x, &#39; ([A-Za-z]+)\.&#39;)&#39;.
 * {Tue Apr 21 22:59:34 2020} [Fillna]: 795.0 missing value(s) of the vcolumn &#34;boat&#34; was/were filled.
 * {Tue Apr 21 22:59:34 2020} [Fillna]: 948.0 missing value(s) of the vcolumn &#34;cabin&#34; was/were filled.
 * {Tue Apr 21 22:59:34 2020} [Drop]: vcolumn &#34;cabin&#34; was deleted from the vDataFrame.
 * {Tue Apr 21 22:59:36 2020} [Eval]: A new vcolumn &#34;family_size&#34; was added to the vDataFrame.
 * {Tue Apr 21 22:59:36 2020} [(CASE WHEN  &lt; -176.6204982585513 THEN -176.6204982585513 WHEN  &gt; 244.5480856064831 THEN 244.5480856064831 ELSE  END)]: The vcolumn &#39;fare&#39; was transformed with the func &#39;x -&gt; (CASE WHEN x &lt; -176.6204982585513 THEN -176.6204982585513 WHEN x &gt; 244.5480856064831 THEN 244.5480856064831 ELSE x END)&#39;.
 * {Tue Apr 21 22:59:36 2020} [Label Encoding]: Label Encoding was applied to the vcolumn &#34;sex&#34; using the following mapping:
	female =&gt; 0	male =&gt; 1
 * {Tue Apr 21 22:59:36 2020} [Fillna]: 237.0 missing value(s) of the vcolumn &#34;age&#34; was/were filled.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">135.6333000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">36.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">134.5000000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">26.5500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">21.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">211.5000000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">50.0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">164.8667000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">45.0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[21]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You already love the Virtual Dataframe, do you? &#128540;</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>If you want to share the object with a member of the team, you can use the following method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">titanic</span><span class="o">.</span><span class="n">to_vdf</span><span class="p">(</span><span class="s2">&quot;titanic&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We created a .vdf file which can be read with the 'read_vdf' function:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="n">read_vdf</span>
<span class="n">titanic2</span> <span class="o">=</span> <span class="n">read_vdf</span><span class="p">(</span><span class="s2">&quot;titanic.vdf&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">titanic2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">135.6333000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">36.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">134.5000000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">26.5500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">21.0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">211.5000000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">50.0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">164.8667000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">45.0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 10
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
<p>Let's now save the vDataframe in the Database to fulfill the next step: Data Modelling.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">drop_view</span>
<span class="n">drop_view</span><span class="p">(</span><span class="s2">&quot;titanic_boat&quot;</span><span class="p">)</span>
<span class="n">drop_view</span><span class="p">(</span><span class="s2">&quot;titanic_no_boat&quot;</span><span class="p">)</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">titanic</span><span class="o">.</span><span class="n">save</span><span class="p">()</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="s2">&quot;boat = 1&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">to_db</span><span class="p">(</span><span class="s2">&quot;titanic_boat&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">load</span><span class="p">(</span>
                 <span class="p">)</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="s2">&quot;boat = 0&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">to_db</span><span class="p">(</span><span class="s2">&quot;titanic_no_boat&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The view titanic_boat was successfully dropped.
The view titanic_no_boat was successfully dropped.
795 element(s) was/were filtered
439 element(s) was/were filtered
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
<h2 id="Machine-Learning">Machine Learning<a class="anchor-link" href="#Machine-Learning">&#182;</a></h2><h3 id="Passengers-with-a-lifeboat">Passengers with a lifeboat<a class="anchor-link" href="#Passengers-with-a-lifeboat">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>First let's look at the number of survivors in this dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataframe</span>
<span class="n">titanic_boat</span> <span class="o">=</span> <span class="n">vDataframe</span><span class="p">(</span><span class="s2">&quot;titanic_boat&quot;</span><span class="p">)</span>
<span class="n">titanic_boat</span><span class="p">[</span><span class="s2">&quot;survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="border: 1px solid white;">"survived"</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>dtype</b></td><td style="border: 1px solid white;">int</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">430</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">9</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We only have 9 death. Let's try to understand why these passengers died.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_boat</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="s2">&quot;survived = 0&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>430 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">21.0000000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">34.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">15.5500000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">30.6958000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">40.9822068965517</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">75.2417000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">36.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">14.4542000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">27.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">15.5500000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">36.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">7.2500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">26.2142058823529</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">7.2500000000000</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">15.8500000000000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">32.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_boat, Number of rows: 9, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>These passengers have no reason to die except the ones in third class. Building a model for this part of the data is useless.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Passengers-without-a-lifeboat">Passengers without a lifeboat<a class="anchor-link" href="#Passengers-without-a-lifeboat">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's now look at passengers without a lifeboat.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataframe</span>
<span class="n">titanic_boat</span> <span class="o">=</span> <span class="n">vDataframe</span><span class="p">(</span><span class="s2">&quot;titanic_no_boat&quot;</span><span class="p">)</span>
<span class="n">titanic_boat</span><span class="p">[</span><span class="s2">&quot;survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>value</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="border: 1px solid white;">"survived"</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>dtype</b></td><td style="border: 1px solid white;">int</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">775</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">20</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[27]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Only 20 survived. Let's look why.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_boat</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="s2">&quot;survived = 1&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">20</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>775 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">28.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">12.6500000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">21.0000000000000</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">14.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.0708000000000</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">42.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">26.0000000000000</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">17.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">10.5000000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">34.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">23.0000000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>6</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">18.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">23.0000000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>7</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">42.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">13.0000000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>8</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">26.2142058823529</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.7500000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>9</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;"> Mr.</td><td style="border: 1px solid white;">26.2142058823529</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.7500000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>10</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">58.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">146.5208000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>11</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">15.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">14.4542000000000</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>12</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">47.0</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.0000000000000</td><td style="border: 1px solid white;">1</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>13</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">22.5766423357664</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.7792000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>14</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8.6833000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>15</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">22.5766423357664</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.2292000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>16</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">27.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.9250000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>17</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">26.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7.9250000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>18</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Miss.</td><td style="border: 1px solid white;">23.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8.0500000000000</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>19</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;"> Mrs.</td><td style="border: 1px solid white;">33.0</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">15.8500000000000</td><td style="border: 1px solid white;">3</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[28]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_no_boat, Number of rows: 20, Number of columns: 10</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>They are mostly women. The famous quotation "Women and children first" is then right. Let's build a model to get more insights.</p>
<p>One of our predictor is categorical (the passenger title). Besides, some of the predictors are correlated. It is preferable to work with a non-linear classifier which can handle that. Random Forest seems to be perfect for the study. Let's evaluate it with a Cross Validation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.model_selection</span> <span class="k">import</span> <span class="n">cross_validate</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">drop_model</span>

<span class="n">predictors</span> <span class="o">=</span> <span class="n">titanic</span><span class="o">.</span><span class="n">get_columns</span><span class="p">(</span><span class="n">exclude_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;&quot;survived&quot;&#39;</span><span class="p">])</span>
<span class="n">response</span> <span class="o">=</span> <span class="s2">&quot;survived&quot;</span>
<span class="n">relation</span> <span class="o">=</span> <span class="s2">&quot;titanic_no_boat&quot;</span>
<span class="n">drop_model</span><span class="p">(</span><span class="s2">&quot;rf_titanic&quot;</span><span class="p">)</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="s2">&quot;rf_titanic&quot;</span><span class="p">,</span> <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">40</span><span class="p">,</span> <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">4</span><span class="p">)</span>
<span class="n">cross_validate</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">relation</span><span class="p">,</span> <span class="n">predictors</span><span class="p">,</span> <span class="n">response</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The model rf_titanic was successfully dropped.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1-fold</b></td><td style="border: 1px solid white;">0.9121428571428566</td><td style="border: 1px solid white;">0.09063019587360353</td><td style="border: 1px solid white;">0.982456140350877</td><td style="border: 1px solid white;">0.0337110933827578</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">-0.01754385964912286</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2-fold</b></td><td style="border: 1px solid white;">0.9555555555555557</td><td style="border: 1px solid white;">0.5280993042378243</td><td style="border: 1px solid white;">0.979838709677419</td><td style="border: 1px solid white;">0.0251294583640899</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">-0.020161290322580627</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3-fold</b></td><td style="border: 1px solid white;">0.9607569721115534</td><td style="border: 1px solid white;">0.5498611498780779</td><td style="border: 1px solid white;">0.961685823754789</td><td style="border: 1px solid white;">0.0429399169028462</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">-0.03831417624521072</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg</b></td><td style="border: 1px solid white;">0.942818461603322</td><td style="border: 1px solid white;">0.38953021666316856</td><td style="border: 1px solid white;">0.9746602245943616</td><td style="border: 1px solid white;">0.0339268228832313</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">-0.02533977540563807</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>std</b></td><td style="border: 1px solid white;">0.026692849516461174</td><td style="border: 1px solid white;">0.25908359906103545</td><td style="border: 1px solid white;">0.011312119328048955</td><td style="border: 1px solid white;">0.008907188824383383</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.011312119328048877</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[29]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As the dataset is unbalanced, the AUC is a good way to evaluate it. <br>
The model is very good with an average greater than 0.9 ! <br>
We can now build a model with the entire dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">relation</span><span class="p">,</span> <span class="n">predictors</span><span class="p">,</span> <span class="n">response</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[30]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>

===========
call_string
===========
SELECT rf_classifier(&#39;public.rf_titanic&#39;, &#39;titanic_no_boat&#39;, &#39;&#34;survived&#34;&#39;, &#39;&#34;boat&#34;, &#34;sibsp&#34;, &#34;fare&#34;, &#34;sex&#34;, &#34;pclass&#34;, &#34;age&#34;, &#34;name&#34;, &#34;parch&#34;, &#34;family_size&#34;&#39; USING PARAMETERS exclude_columns=&#39;&#39;, ntree=40, mtry=4, sampling_size=0.632, max_depth=4, max_breadth=1000000000, min_leaf_size=1, min_info_gain=0, nbins=32);

=======
details
=======
 predictor |      type      
-----------+----------------
   boat    |      int       
   sibsp   |      int       
   fare    |float or numeric
    sex    |      int       
  pclass   |      int       
    age    |float or numeric
   name    |char or varchar 
   parch   |      int       
family_size|      int       


===============
Additional Info
===============
       Name       |Value
------------------+-----
    tree_count    | 40  
rejected_row_count|  1  
accepted_row_count| 794 </pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's look at the features importance.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">features_importance</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>importance</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="border: 1px solid white;">24.74</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="border: 1px solid white;">16.43</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="border: 1px solid white;">14.27</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="border: 1px solid white;">12.13</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="border: 1px solid white;">10.34</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="border: 1px solid white;">8.54</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td><td style="border: 1px solid white;">7.63</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>family_size</b></td><td style="border: 1px solid white;">5.91</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="border: 1px solid white;">0.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[31]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As expected, the title and the sex are the most important predictors.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion">&#182;</a></h2><p>We have solved this use-case in a pandas-like way but we never loaded the data in memory. This example showed an overview of the library. You can now start your own project by looking at the documentation first.</p>

</div>
</div>
</div>
