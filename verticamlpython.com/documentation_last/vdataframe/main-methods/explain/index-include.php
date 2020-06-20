<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.explain">vDataFrame.explain<a class="anchor-link" href="#vDataFrame.explain">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">explain</span><span class="p">(</span><span class="n">digraph</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Provides information on how Vertica is computing the current vDataFrame relation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">digraph</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, returns only the digraph of the explain plan.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>str</b> : Explain Plan.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[90]:</div>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>parch</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.0</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.0</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.5042</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.0</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[91]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Doing some operations</span>
<span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">label_encode</span><span class="p">()</span>
<span class="n">titanic</span><span class="p">[</span><span class="s2">&quot;embarked&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">()</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">fillna</span><span class="p">()</span>
<span class="c1"># Explaining the vDataFrame current relation</span>
<span class="nb">print</span><span class="p">(</span><span class="n">titanic</span><span class="o">.</span><span class="n">explain</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>------------------------------ 
QUERY PLAN DESCRIPTION: 

EXPLAIN SELECT * FROM (SELECT &#34;survived&#34;, COALESCE(&#34;boat&#34;, &#39;13&#39;) AS &#34;boat&#34;, &#34;ticket&#34;, COALESCE(&#34;embarked&#34;, &#39;S&#39;) AS &#34;embarked&#34;, COALESCE(&#34;home.dest&#34;, &#39;New York, NY&#39;) AS &#34;home.dest&#34;, &#34;sibsp&#34;, COALESCE(&#34;fare&#34;, 33.9637936739659) AS &#34;fare&#34;, DECODE(&#34;sex&#34;, &#39;female&#39;, 0, &#39;male&#39;, 1, 2) AS &#34;sex&#34;, COALESCE(&#34;body&#34;, 164.14406779661) AS &#34;body&#34;, &#34;pclass&#34;, COALESCE(&#34;age&#34;, 30.1524573721163) AS &#34;age&#34;, &#34;name&#34;, COALESCE(&#34;cabin&#34;, &#39;C23 C25 C27&#39;) AS &#34;cabin&#34;, &#34;parch&#34;, DECODE(&#34;embarked_C&#34;, &#39;C&#39;, 1, 0) AS &#34;embarked_C&#34;, DECODE(&#34;embarked_Q&#34;, &#39;Q&#39;, 1, 0) AS &#34;embarked_Q&#34; FROM (SELECT &#34;survived&#34;, &#34;boat&#34;, &#34;ticket&#34;, &#34;embarked&#34;, &#34;home.dest&#34;, &#34;sibsp&#34;, &#34;fare&#34;, &#34;sex&#34;, &#34;body&#34;, &#34;pclass&#34;, &#34;age&#34;, &#34;name&#34;, &#34;cabin&#34;, &#34;parch&#34;, &#34;embarked&#34; AS &#34;embarked_C&#34;, &#34;embarked&#34; AS &#34;embarked_Q&#34; FROM &#34;public&#34;.&#34;titanic&#34;) VERTICA_ML_PYTHON_SUBTABLE) VERTICA_ML_PYTHON_SUBTABLE

Access Path:
+-STORAGE ACCESS for titanic [Cost: 96, Rows: 1K (NO STATISTICS)] (PATH ID: 1)
|  Projection: public.titanic_super
|  Materialize: titanic.pclass, titanic.survived, titanic.name, titanic.sex, titanic.age, titanic.sibsp, titanic.parch, titanic.ticket, titanic.fare, titanic.cabin, titanic.embarked, titanic.boat, titanic.body, titanic.&#34;home.dest&#34;


----------------------------------------------- 
PLAN: BASE QUERY PLAN (GraphViz Format)
----------------------------------------------- 
digraph G {
graph [rankdir=BT, label = &#34;BASE QUERY PLAN
	Query: EXPLAIN SELECT * FROM (SELECT \&#34;survived\&#34;, COALESCE(\&#34;boat\&#34;, \&#39;13\&#39;) AS \&#34;boat\&#34;, \&#34;ticket\&#34;, COALESCE(\&#34;embarked\&#34;, \&#39;S\&#39;) AS \&#34;embarked\&#34;, COALESCE(\&#34;home.dest\&#34;, \&#39;New York, NY\&#39;) AS \&#34;home.dest\&#34;, \&#34;sibsp\&#34;, COALESCE(\&#34;fare\&#34;, 33.9637936739659) AS \&#34;fare\&#34;, DECODE(\&#34;sex\&#34;, \&#39;female\&#39;, 0, \&#39;male\&#39;, 1, 2) AS \&#34;sex\&#34;, COALESCE(\&#34;body\&#34;, 164.14406779661) AS \&#34;body\&#34;, \&#34;pclass\&#34;, COALESCE(\&#34;age\&#34;, 30.1524573721163) AS \&#34;age\&#34;, \&#34;name\&#34;, COALESCE(\&#34;cabin\&#34;, \&#39;C23 C25 C27\&#39;) AS \&#34;cabin\&#34;, \&#34;parch\&#34;, DECODE(\&#34;embarked_C\&#34;, \&#39;C\&#39;, 1, 0) AS \&#34;embarked_C\&#34;, DECODE(\&#34;embarked_Q\&#34;, \&#39;Q\&#39;, 1, 0) AS \&#34;embarked_Q\&#34; FROM (SELECT \&#34;survived\&#34;, \&#34;boat\&#34;, \&#34;ticket\&#34;, \&#34;embarked\&#34;, \&#34;home.dest\&#34;, \&#34;sibsp\&#34;, \&#34;fare\&#34;, \&#34;sex\&#34;, \&#34;body\&#34;, \&#34;pclass\&#34;, \&#34;age\&#34;, \&#34;name\&#34;, \&#34;cabin\&#34;, \&#34;parch\&#34;, \&#34;embarked\&#34; AS \&#34;embarked_C\&#34;, \&#34;embarked\&#34; AS \&#34;embarked_Q\&#34; FROM \&#34;public\&#34;.\&#34;titanic\&#34;) VERTICA_ML_PYTHON_SUBTABLE) VERTICA_ML_PYTHON_SUBTABLE
	
	All Nodes Vector: 
	
	  node[0]=v_testdb_node0001 (initiator) Up
	&#34;, labelloc=t, labeljust=l ordering=out]
0[label = &#34;Root 
	OutBlk=[UncTuple(16)]&#34;, color = &#34;green&#34;, shape = &#34;house&#34;];
1[label = &#34;NewEENode 
	OutBlk=[UncTuple(16)]&#34;, color = &#34;green&#34;, shape = &#34;box&#34;];
2[label = &#34;StorageUnionStep: titanic_super
	Unc: Integer(8)
	Unc: Varchar(100)
	Unc: Varchar(36)
	Unc: Varchar(20)
	Unc: Varchar(100)
	Unc: Integer(8)
	Unc: Numeric(18, 13)
	Unc: Integer(8)
	Unc: Numeric(29, 11)
	Unc: Integer(8)
	Unc: Numeric(16, 13)
	Unc: Varchar(164)
	Unc: Varchar(30)
	Unc: Integer(8)
	Unc: Integer(8)
	Unc: Integer(8)&#34;, color = &#34;purple&#34;, shape = &#34;box&#34;];
3[label = &#34;ExprEval: 
	  titanic.survived
	  coalesce(titanic.boat, \&#39;13\&#39;)
	  titanic.ticket
	  coalesce(titanic.embarked, \&#39;S\&#39;)
	  coalesce(titanic.\&#34;home.dest\&#34;, \&#39;New York, NY\&#39;)
	  titanic.sibsp
	  coalesce(titanic.fare, 33.9637936739659)
	  CASE titanic.sex WHEN NULLSEQUAL \&#39;female\&#39; THEN 0 WHEN NULLSEQUAL \&#39;male\&#39; THEN 1 ELSE 2 END
	  coalesce(titanic.body, 164.14406779661)
	  titanic.pclass
	  coalesce(titanic.age, 30.1524573721163)
	  titanic.name
	  coalesce(titanic.cabin, \&#39;C23 C25 C27\&#39;)
	  titanic.parch
	  CASE titanic.embarked WHEN NULLSEQUAL \&#39;C\&#39; THEN 1 ELSE 0 END
	  CASE titanic.embarked WHEN NULLSEQUAL \&#39;Q\&#39; THEN 1 ELSE 0 END
	Unc: Integer(8)
	Unc: Varchar(100)
	Unc: Varchar(36)
	Unc: Varchar(20)
	Unc: Varchar(100)
	Unc: Integer(8)
	Unc: Numeric(18, 13)
	Unc: Integer(8)
	Unc: Numeric(29, 11)
	Unc: Integer(8)
	Unc: Numeric(16, 13)
	Unc: Varchar(164)
	Unc: Varchar(30)
	Unc: Integer(8)
	Unc: Integer(8)
	Unc: Integer(8)&#34;, color = &#34;brown&#34;, shape = &#34;box&#34;];
4[label = &#34;ScanStep: titanic_super
	pclass
	survived
	name
	sex
	age
	sibsp
	parch
	ticket
	fare
	cabin
	embarked
	boat
	body
	home.dest
	Unc: Integer(8)
	Unc: Integer(8)
	Unc: Varchar(164)
	Unc: Varchar(20)
	Unc: Numeric(6, 3)
	Unc: Integer(8)
	Unc: Integer(8)
	Unc: Varchar(36)
	Unc: Numeric(10, 5)
	Unc: Varchar(30)
	Unc: Varchar(20)
	Unc: Varchar(100)
	Unc: Integer(8)
	Unc: Varchar(100)&#34;, color = &#34;brown&#34;, shape = &#34;box&#34;];
1-&gt;0 [label = &#34;V[0] C=16&#34;, color = &#34;black&#34;, style=&#34;bold&#34;, arrowtail=&#34;inv&#34;];
2-&gt;1 [label = &#34;0&#34;, color = &#34;blue&#34;];
3-&gt;2 [label = &#34;0&#34;, color = &#34;blue&#34;];
4-&gt;3 [label = &#34;0&#34;, color = &#34;blue&#34;];}
</pre>
</div>
</div>

</div>
</div>

</div>