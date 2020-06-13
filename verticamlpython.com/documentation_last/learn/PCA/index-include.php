<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="PCA">PCA<a class="anchor-link" href="#PCA">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">PCA</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
    <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
    <span class="n">n_components</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span>
    <span class="n">scale</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">,</span> 
    <span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;lapack&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a PCA (Principal Component Analysis) object by using the Vertica Highly Distributed and Scalable PCA on the data.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the the model. The model will be stored in the DB.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">n_components</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The number of components to keep in the model. If this value is not provided, all components are kept. The maximum number of components is the number of non-zero singular values returned by the internal call to SVD. This number is less than or equal to SVD (number of columns, number of rows). </td> </tr>
    <tr> <td><div class="param_name">scale</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>A Boolean value that specifies whether to standardize the columns during the preparation step.</td> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>The method used to calculate PCA. <br><ul><li><b>method</b> : Lapack definition.</li></ul></td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">components</div></td> <td><div class="type">tablesample</div></td> <td>The principal components.</td> </tr>
    <tr> <td><div class="param_name">explained_variance</div></td> <td><div class="type">tablesample</div></td> <td>The singular values explained variance.</td> </tr>
    <tr> <td><div class="param_name">mean</div></td> <td><div class="type">tablesample</div></td> <td>The information about columns from the input relation used for creating the PCA model.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Unsupervised/deploySQL2/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/deployInverseSQL/index.php">deployInverseSQL</a></td> <td>Returns the SQL code needed to deploy the inverse model (PCA ** -1).</td> </tr>
    <tr> <td><a href="../Unsupervised/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Unsupervised/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/to_vdf3/index.php">to_vdf</a></td> <td>Creates a vDataFrame of the model.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.decomposition</span> <span class="k">import</span> <span class="n">PCA</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">PCA</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.pca_iris&quot;</span><span class="p">)</span>
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
<pre>&lt;PCA&gt;
</pre>
</div>
</div>

</div>
</div>

</div>