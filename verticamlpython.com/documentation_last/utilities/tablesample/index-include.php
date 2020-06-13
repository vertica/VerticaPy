<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="tablesample">tablesample<a class="anchor-link" href="#tablesample">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">tablesample</span><span class="p">(</span><span class="n">values</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{},</span> 
            <span class="n">dtype</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{},</span> 
            <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;Sample&quot;</span><span class="p">,</span>
            <span class="n">count</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span> 
            <span class="n">offset</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span> 
            <span class="n">table_info</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The tablesample is the transition from 'Big Data' to 'Small Data'. This object was created to have a nice way of displaying the results and to not have any dependency to any other module. It stores the aggregated result in memory and has some useful method to transform it to pandas.DataFrame or vDataFrame.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">values</div></td> <td><div class="type">dict</div></td> <td><div class = "yes">&#10003;</div></td> <td>Dictionary of columns (keys) and their values. The dictionary must be similar to the following one: {"column1": [val1, ..., valm], ... "columnk": [val1, ..., valm]}</td> </tr>
    <tr> <td><div class="param_name">dtype</div></td> <td><div class="type">dict</div></td> <td><div class = "yes">&#10003;</div></td> <td>Columns data types.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the object. It is used only for rendering purposes.</td> </tr>
    <tr> <td><div class="param_name">count</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of elements if we had to load the entire dataset. It is used only for rendering purposes.</td> </tr>
    <tr> <td><div class="param_name">offset</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of elements which had been skipped if we had to load the entire dataset. It is used only for rendering purposes.</td> </tr>
    <tr> <td><div class="param_name">table_info</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, the tablesample informations will be displayed.</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>The tablesample attributes are the same than the parameters.</p>
<h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="transpose">transpose</a></td> <td>Transposes the tablesample.</td> </tr>
    <tr> <td><a href="to_pandas">to_pandas</a></td> <td>Converts the tablesample to a pandas DataFrame.</td> </tr>
    <tr> <td><a href="to_sql">to_sql</a></td> <td>Generates the SQL query associated to the tablesample.</td> </tr>
    <tr> <td><a href="to_vdf">to_vdf</a></td> <td>Converts the tablesample to a vDataFrame.</td> </tr>
</table>
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
<span class="n">dataset</span> <span class="o">=</span> <span class="n">tablesample</span><span class="p">(</span><span class="n">values</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;index&quot;</span><span class="p">:</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span>
                                <span class="s2">&quot;name&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Badr&quot;</span><span class="p">,</span> <span class="s2">&quot;Fouad&quot;</span><span class="p">,</span> <span class="s2">&quot;Colin&quot;</span><span class="p">],</span>
                                <span class="s2">&quot;first_name&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Ouali&quot;</span><span class="p">,</span> <span class="s2">&quot;Teban&quot;</span><span class="p">,</span> <span class="s2">&quot;Mahony&quot;</span><span class="p">]},</span>
                      <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;Vertica Team&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">dataset</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>first_name</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Badr</td><td style="border: 1px solid white;">Ouali</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Fouad</td><td style="border: 1px solid white;">Teban</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Colin</td><td style="border: 1px solid white;">Mahony</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: Vertica Team, Number of rows: 3, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>