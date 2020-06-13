<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="LocalOutlierFactor">LocalOutlierFactor (Beta)<a class="anchor-link" href="#LocalOutlierFactor">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">LocalOutlierFactor</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                   <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
                   <span class="n">n_neighbors</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> 
                   <span class="n">p</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a LocalOutlierFactor object by using the Local Outlier Factor algorithm as defined by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg Sander. This object is using pure SQL to compute all the distances and final score. It is using CROSS JOIN and may be really expensive in some cases. It will index all the elements of the table in order to be optimal (the CROSS JOIN will happen only with IDs which are integers). As LocalOutlierFactor is using the p-distance, it is highly sensible to un-normalized data.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the the model. As it is not a built in model, this name will be used to build the final table.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">n_neighbors</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of neighbors to consider when computing the score.</td> </tr>
    <tr> <td><div class="param_name">p</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The p of the p-distance (distance metric used during the model computation).</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">n_errors</div></td> <td><div class="type">int</div></td> <td>Number of errors during the LOF computation.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">key_columns</div></td> <td><div class="type">list</div></td> <td>Columns not used during the algorithm computation but which will be used to create the final relation.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Unsupervised/fit2/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/info/index.php">info</a></td> <td>Displays some information about the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/plot2/index.php">plot</a></td> <td>Draws the model if the number of predictors is 2 or 3.</td> </tr>
    <tr> <td><a href="../Unsupervised/to_vdf/index.php">to_vdf</a></td> <td>Creates a vDataFrame of the model.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.neighbors</span> <span class="k">import</span> <span class="n">LocalOutlierFactor</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">LocalOutlierFactor</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.LOF_heart&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;LocalOutlierFactor&gt;
</pre>
</div>
</div>

</div>
</div>

</div>