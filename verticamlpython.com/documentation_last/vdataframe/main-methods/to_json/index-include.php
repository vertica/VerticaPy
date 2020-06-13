<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.to_json">vDataFrame.to_json<a class="anchor-link" href="#vDataFrame.to_json">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">to_json</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                   <span class="n">path</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span> 
                   <span class="n">usecols</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                   <span class="n">order_by</span> <span class="o">=</span> <span class="p">[],</span>
                   <span class="n">limit</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a JSON file of the current vDataFrame relation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the JSON file. Be careful: if a JSON file with the same name exists, it will over-write it.</td> </tr>
    <tr> <td><div class="param_name">path</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Absolute path where the JSON file will be created.</td> </tr>
    <tr> <td><div class="param_name">usecols</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vcolumns to select from the final vDataFrame relation. If empty, all the vcolumns will be selected.</td> </tr>
    <tr> <td><div class="param_name">order_by</div></td> <td><div class="type">dict / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the vcolumns used to sort the data using asc order or dictionary of all the sorting methods. For example, to sort by "column1" ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}</td> </tr>
    <tr> <td><div class="param_name">limit</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>If greater than 0, the maximum number of elements to write at the same time in the JSON file. It can be used to minimize memory impacts. Be sure to keep the same order to avoid unexpected results.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.000</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">30.000</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">151.55000</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">25.000</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.00000</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">39.000</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">0</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">49.50420</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">22</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">71.000</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">0</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Doing some transformations</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">()</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">normalize</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>survived</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>boat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>ticket</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>home.dest</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>body</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>age</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>cabin</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked_C</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked_Q</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp_0</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp_1</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp_2</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp_3</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp_4</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sibsp_5</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>sex_female</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass_1</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass_2</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_0</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_1</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_2</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_3</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_4</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_5</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>parch_6</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-0.757307371153963162979989569746</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">0.476361452530317344428951316723</td><td style="border: 1px solid white;">2.2335228377568673306163744003</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">-1.52458485653939825566284767748</td><td style="border: 1px solid white;">-1.9502503129565278908367342155</td><td style="border: 1px solid white;">Allison, Miss. Helen Loraine</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">1.866851401077760435621662793693</td><td style="border: 1px solid white;">-0.507632846372679881779074245706</td><td style="border: 1px solid white;">-0.30642369222116871970059157132</td><td style="border: 1px solid white;">-1.448723687429805724921492057367</td><td style="border: 1px solid white;">1.74835105897043756352888762989</td><td style="border: 1px solid white;">-0.1781763946794669298088450104197</td><td style="border: 1px solid white;">-0.1283008734279738857812014957876</td><td style="border: 1px solid white;">-0.1346740711671781651944719413214</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">1.391590693589806938988264634593</td><td style="border: 1px solid white;">1.718351957456489070218766430323</td><td style="border: 1px solid white;">-0.515194873214845095325784951282</td><td style="border: 1px solid white;">-1.82404116243169099688897026934</td><td style="border: 1px solid white;">-0.389962660998658638813196185979</td><td style="border: 1px solid white;">3.4032988446260016658518137173233</td><td style="border: 1px solid white;">-0.080746501890544571964712271643787</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.040274819467791498027366578185165</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">-0.757307371153963162979989569746</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">0.476361452530317344428951316723</td><td style="border: 1px solid white;">2.2335228377568673306163744003</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">-0.30177333429701812091999864</td><td style="border: 1px solid white;">-1.52458485653939825566284767748</td><td style="border: 1px solid white;">-0.0105614239550127329564098413</td><td style="border: 1px solid white;">Allison, Mr. Hudson Joshua Creighton</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">1.866851401077760435621662793693</td><td style="border: 1px solid white;">-0.507632846372679881779074245706</td><td style="border: 1px solid white;">-0.30642369222116871970059157132</td><td style="border: 1px solid white;">-1.448723687429805724921492057367</td><td style="border: 1px solid white;">1.74835105897043756352888762989</td><td style="border: 1px solid white;">-0.1781763946794669298088450104197</td><td style="border: 1px solid white;">-0.1283008734279738857812014957876</td><td style="border: 1px solid white;">-0.1346740711671781651944719413214</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.718019768191301048660149415335</td><td style="border: 1px solid white;">1.718351957456489070218766430323</td><td style="border: 1px solid white;">-0.515194873214845095325784951282</td><td style="border: 1px solid white;">-1.82404116243169099688897026934</td><td style="border: 1px solid white;">-0.389962660998658638813196185979</td><td style="border: 1px solid white;">3.4032988446260016658518137173233</td><td style="border: 1px solid white;">-0.080746501890544571964712271643787</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.040274819467791498027366578185165</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">-0.757307371153963162979989569746</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">113781</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Montreal, PQ / Chesterville, ON</td><td style="border: 1px solid white;">0.476361452530317344428951316723</td><td style="border: 1px solid white;">2.2335228377568673306163744003</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">-1.52458485653939825566284767748</td><td style="border: 1px solid white;">-0.3569344398481404397207534796</td><td style="border: 1px solid white;">Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td style="border: 1px solid white;">C22 C26</td><td style="border: 1px solid white;">1.866851401077760435621662793693</td><td style="border: 1px solid white;">-0.507632846372679881779074245706</td><td style="border: 1px solid white;">-0.30642369222116871970059157132</td><td style="border: 1px solid white;">-1.448723687429805724921492057367</td><td style="border: 1px solid white;">1.74835105897043756352888762989</td><td style="border: 1px solid white;">-0.1781763946794669298088450104197</td><td style="border: 1px solid white;">-0.1283008734279738857812014957876</td><td style="border: 1px solid white;">-0.1346740711671781651944719413214</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">1.391590693589806938988264634593</td><td style="border: 1px solid white;">1.718351957456489070218766430323</td><td style="border: 1px solid white;">-0.515194873214845095325784951282</td><td style="border: 1px solid white;">-1.82404116243169099688897026934</td><td style="border: 1px solid white;">-0.389962660998658638813196185979</td><td style="border: 1px solid white;">3.4032988446260016658518137173233</td><td style="border: 1px solid white;">-0.080746501890544571964712271643787</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.040274819467791498027366578185165</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-0.757307371153963162979989569746</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">112050</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">Belfast, NI</td><td style="border: 1px solid white;">-0.484145136395191051224867122640</td><td style="border: 1px solid white;">-0.6451344183800711827483251581</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">-1.52458485653939825566284767748</td><td style="border: 1px solid white;">0.6129100046526171392194087075</td><td style="border: 1px solid white;">Andrews, Mr. Thomas Jr</td><td style="border: 1px solid white;">A36</td><td style="border: 1px solid white;">-0.435691956173569945178825405433</td><td style="border: 1px solid white;">-0.507632846372679881779074245706</td><td style="border: 1px solid white;">-0.30642369222116871970059157132</td><td style="border: 1px solid white;">0.689703382293138372371557977476</td><td style="border: 1px solid white;">-0.57150400207205686533461107422</td><td style="border: 1px solid white;">-0.1781763946794669298088450104197</td><td style="border: 1px solid white;">-0.1283008734279738857812014957876</td><td style="border: 1px solid white;">-0.1346740711671781651944719413214</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.718019768191301048660149415335</td><td style="border: 1px solid white;">1.718351957456489070218766430323</td><td style="border: 1px solid white;">-0.515194873214845095325784951282</td><td style="border: 1px solid white;">0.54778896869655520601485651247</td><td style="border: 1px solid white;">-0.389962660998658638813196185979</td><td style="border: 1px solid white;">-0.2935944425821727290366499712179</td><td style="border: 1px solid white;">-0.080746501890544571964712271643787</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.040274819467791498027366578185165</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">-0.757307371153963162979989569746</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">PC 17609</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">Montevideo, Uruguay</td><td style="border: 1px solid white;">-0.484145136395191051224867122640</td><td style="border: 1px solid white;">0.2951864297839290254556257484</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">-1.47183603843091257442431516</td><td style="border: 1px solid white;">-1.52458485653939825566284767748</td><td style="border: 1px solid white;">2.8296973063686344625112079923</td><td style="border: 1px solid white;">Artagaveytia, Mr. Ramon</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">-0.435691956173569945178825405433</td><td style="border: 1px solid white;">1.968331313405532477147732836418</td><td style="border: 1px solid white;">-0.30642369222116871970059157132</td><td style="border: 1px solid white;">0.689703382293138372371557977476</td><td style="border: 1px solid white;">-0.57150400207205686533461107422</td><td style="border: 1px solid white;">-0.1781763946794669298088450104197</td><td style="border: 1px solid white;">-0.1283008734279738857812014957876</td><td style="border: 1px solid white;">-0.1346740711671781651944719413214</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.718019768191301048660149415335</td><td style="border: 1px solid white;">1.718351957456489070218766430323</td><td style="border: 1px solid white;">-0.515194873214845095325784951282</td><td style="border: 1px solid white;">0.54778896869655520601485651247</td><td style="border: 1px solid white;">-0.389962660998658638813196185979</td><td style="border: 1px solid white;">-0.2935944425821727290366499712179</td><td style="border: 1px solid white;">-0.080746501890544571964712271643787</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.069871553648533438650839486531736</td><td style="border: 1px solid white;">-0.040274819467791498027366578185165</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[2]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic, Number of rows: 1234, Number of columns: 32</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Creating the JSON file of the current vDataFrame</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">to_json</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;titanic&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Reading the JSON file</span>
<span class="n">file</span> <span class="o">=</span> <span class="nb">open</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">,</span> <span class="s2">&quot;r&quot;</span><span class="p">)</span> 
<span class="n">file</span><span class="o">.</span><span class="n">readline</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;line1:&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">file</span><span class="o">.</span><span class="n">readline</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;line2:&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">file</span><span class="o">.</span><span class="n">readline</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;...&quot;</span><span class="p">)</span>
<span class="n">file</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>line1:
{&#34;survived&#34;: -0.757307371153963162979989569746, &#34;ticket&#34;: &#34;113781&#34;, &#34;embarked&#34;: &#34;S&#34;, &#34;home.dest&#34;: &#34;Montreal, PQ / Chesterville, ON&#34;, &#34;sibsp&#34;: 0.476361452530317344428951316723, &#34;fare&#34;: 2.2335228377568673306163744003, &#34;sex&#34;: &#34;female&#34;, &#34;pclass&#34;: -1.52458485653939825566284767748, &#34;age&#34;: -1.9502503129565278908367342155, &#34;name&#34;: &#34;Allison, Miss. Helen Loraine&#34;, &#34;cabin&#34;: &#34;C22 C26&#34;, &#34;parch&#34;: 1.866851401077760435621662793693, &#34;embarked_C&#34;: -0.507632846372679881779074245706, &#34;embarked_Q&#34;: -0.30642369222116871970059157132, &#34;sibsp_0&#34;: -1.448723687429805724921492057367, &#34;sibsp_1&#34;: 1.74835105897043756352888762989, &#34;sibsp_2&#34;: -0.1781763946794669298088450104197, &#34;sibsp_3&#34;: -0.1283008734279738857812014957876, &#34;sibsp_4&#34;: -0.1346740711671781651944719413214, &#34;sibsp_5&#34;: -0.069871553648533438650839486531736, &#34;sex_female&#34;: 1.391590693589806938988264634593, &#34;pclass_1&#34;: 1.718351957456489070218766430323, &#34;pclass_2&#34;: -0.515194873214845095325784951282, &#34;parch_0&#34;: -1.82404116243169099688897026934, &#34;parch_1&#34;: -0.389962660998658638813196185979, &#34;parch_2&#34;: 3.4032988446260016658518137173233, &#34;parch_3&#34;: -0.080746501890544571964712271643787, &#34;parch_4&#34;: -0.069871553648533438650839486531736, &#34;parch_5&#34;: -0.069871553648533438650839486531736, &#34;parch_6&#34;: -0.040274819467791498027366578185165},

line2:
{&#34;survived&#34;: -0.757307371153963162979989569746, &#34;ticket&#34;: &#34;113781&#34;, &#34;embarked&#34;: &#34;S&#34;, &#34;home.dest&#34;: &#34;Montreal, PQ / Chesterville, ON&#34;, &#34;sibsp&#34;: 0.476361452530317344428951316723, &#34;fare&#34;: 2.2335228377568673306163744003, &#34;sex&#34;: &#34;male&#34;, &#34;body&#34;: -0.30177333429701812091999864, &#34;pclass&#34;: -1.52458485653939825566284767748, &#34;age&#34;: -0.0105614239550127329564098413, &#34;name&#34;: &#34;Allison, Mr. Hudson Joshua Creighton&#34;, &#34;cabin&#34;: &#34;C22 C26&#34;, &#34;parch&#34;: 1.866851401077760435621662793693, &#34;embarked_C&#34;: -0.507632846372679881779074245706, &#34;embarked_Q&#34;: -0.30642369222116871970059157132, &#34;sibsp_0&#34;: -1.448723687429805724921492057367, &#34;sibsp_1&#34;: 1.74835105897043756352888762989, &#34;sibsp_2&#34;: -0.1781763946794669298088450104197, &#34;sibsp_3&#34;: -0.1283008734279738857812014957876, &#34;sibsp_4&#34;: -0.1346740711671781651944719413214, &#34;sibsp_5&#34;: -0.069871553648533438650839486531736, &#34;sex_female&#34;: -0.718019768191301048660149415335, &#34;pclass_1&#34;: 1.718351957456489070218766430323, &#34;pclass_2&#34;: -0.515194873214845095325784951282, &#34;parch_0&#34;: -1.82404116243169099688897026934, &#34;parch_1&#34;: -0.389962660998658638813196185979, &#34;parch_2&#34;: 3.4032988446260016658518137173233, &#34;parch_3&#34;: -0.080746501890544571964712271643787, &#34;parch_4&#34;: -0.069871553648533438650839486531736, &#34;parch_5&#34;: -0.069871553648533438650839486531736, &#34;parch_6&#34;: -0.040274819467791498027366578185165},

...
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
<h3 id="See-Also">See Also<a class="anchor-link" href="#See-Also">&#182;</a></h3><table id="seealso">
    <tr><td><a href="../to_csv">vDataFrame.to_csv</a></td> <td>Creates a csv file of the current vDataFrame relation.</td></tr>
    <tr><td><a href="../to_db">vDataFrame.to_db</a></td> <td>Saves the vDataFrame current relation to the Vertica Database.</td></tr>
</table>
</div>
</div>
</div>