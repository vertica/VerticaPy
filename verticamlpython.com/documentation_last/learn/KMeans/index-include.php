<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="KMeans">KMeans<a class="anchor-link" href="#KMeans">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">KMeans</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
       <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
       <span class="n">n_cluster</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">8</span><span class="p">,</span>
       <span class="n">init</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;kmeanspp&quot;</span><span class="p">,</span>
       <span class="n">max_iter</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">300</span><span class="p">,</span>
       <span class="n">tol</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a KMeans object by using the Vertica Highly Distributed and Scalable KMeans on the data. K-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the model to be stored in the database.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">n_cluster</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of clusters</td> </tr>
    <tr> <td><div class="param_name">init</div></td> <td><div class="type">str / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>The method used to find the initial cluster centers. <br> <ul><li><b>kmeanspp</b> : Uses the KMeans++ method to initialize the centers.</li><li><b>random</b> : The initial centers.</li></ul>It can be also a list with the initial cluster centers to use.</td> </tr>
    <tr> <td><div class="param_name">max_iter</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum number of iterations the algorithm performs.</td> </tr>
    <tr> <td><div class="param_name">tol</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Determines whether the algorithm has converged. The algorithm is considered converged after no center has moved more than a distance of 'tol' from the previous iteration. </td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">cluster_centers</div></td> <td><div class="type">tablesample</div></td> <td>Clusters result of the algorithm.</td> </tr>
    <tr> <td><div class="param_name">metrics</div></td> <td><div class="type">tablesample</div></td> <td>Different metrics to evaluate the model.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Unsupervised/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Unsupervised/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/plot/index.php">plot</a></td> <td>Draws the KMeans clusters.</td> </tr>
    <tr> <td><a href="../Unsupervised/predict/index.php">predict</a></td> <td>Adds the prediction in a vDataFrame.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.cluster</span> <span class="k">import</span> <span class="n">KMeans</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KMeans</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.kmeans_iris&quot;</span><span class="p">,</span>
               <span class="n">n_cluster</span> <span class="o">=</span> <span class="mi">8</span><span class="p">,</span>
               <span class="n">init</span> <span class="o">=</span> <span class="s2">&quot;kmeanspp&quot;</span><span class="p">,</span>
               <span class="n">max_iter</span> <span class="o">=</span> <span class="mi">300</span><span class="p">,</span>
               <span class="n">tol</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">)</span>
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
<pre>&lt;KMeans&gt;
</pre>
</div>
</div>

</div>
</div>

</div>