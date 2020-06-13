<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vdf_from_relation">vdf_from_relation<a class="anchor-link" href="#vdf_from_relation">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf_from_relation</span><span class="p">(</span><span class="n">relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                  <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;VDF&quot;</span><span class="p">,</span> 
                  <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
                  <span class="n">dsn</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span> 
                  <span class="n">schema</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
                  <span class="n">schema_writing</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
                  <span class="n">history</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">saving</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
                  <span class="n">query_on</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span>
                  <span class="n">time_on</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a vDataFrame based on a customized relation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Relation. It can be a customized relation but you need to englobe it using an alias. For example "(SELECT 1) x" is correct whereas "(SELECT 1)" or "SELECT 1" are incorrect.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the vDataFrame. It is used only when displaying the vDataFrame.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">dsn</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Data Base DSN. OS File including the DB credentials. Vertica ML Python will try to create a vertica_python cursor first. If it didn't find the library, it will try to create a pyodbc cursor. Check out utilities.vHelp, it may help you.</td> </tr>
    <tr> <td><div class="param_name">schema</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Relation schema. It can be used to be less ambiguous and allow to create schema and relation name with dots '.' inside.</td> </tr>
    <tr> <td><div class="param_name">schema_writing</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Schema used to create the temporary table. If empty, the function will create a local temporary table.</td> </tr>
    <tr> <td><div class="param_name">history</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vDataFrame history (user modifications). Used to keep the previous vDataFrame history.</td> </tr>
    <tr> <td><div class="param_name">saving</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List used to reconstruct the vDataFrame from previous transformations.</td> </tr>
    <tr> <td><div class="param_name">query_on</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, all the query will be printed.</td> </tr>
    <tr> <td><div class="param_name">time_on</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, all the query elapsed time will be printed.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : The vDataFrame associated to the input relation.</p>

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
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">vdf_from_relation</span><span class="p">(</span><span class="s2">&quot;(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>pclass</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>embarked</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>AVG</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">0.318681318681319</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">Q</td><td style="border: 1px solid white;">0.260416666666667</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0.203781512605042</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">C</td><td style="border: 1px solid white;">0.681481481481481</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">Q</td><td style="border: 1px solid white;">0.666666666666667</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: vdf, Number of rows: 10, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>