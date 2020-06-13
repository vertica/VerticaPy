<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="CountVectorizer">CountVectorizer<a class="anchor-link" href="#CountVectorizer">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">CountVectorizer</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                <span class="n">lowercase</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
                <span class="n">max_df</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span>
                <span class="n">min_df</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                <span class="n">max_features</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span>
                <span class="n">ignore_special</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
                <span class="n">max_text_size</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">2000</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a Text Index which will count the occurences of each word in the data.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the the model. The model will be stored in the DB.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">lowercase</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>Converts all the elements to lowercase before processing.</td> </tr>
    <tr> <td><div class="param_name">max_df</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Keeps the words which represent less than this float in the total dictionary distribution.</td> </tr>
    <tr> <td><div class="param_name">min_df</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Keeps the words which represent more than this float in the total dictionary distribution.</td> </tr>
    <tr> <td><div class="param_name">max_features</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Keeps only the top words of the dictionary.</td> </tr>
    <tr> <td><div class="param_name">ignore_special</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>Ignores all the special characters to build the dictionary.</td> </tr>
    <tr> <td><div class="param_name">max_text_size</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum size of the column which is the concatenation of all the text columns during the fitting.</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">stop_words</div></td> <td><div class="type">list</div></td> <td>The words not added to the vocabulary.</td> </tr>
    <tr> <td><div class="param_name">vocabulary</div></td> <td><div class="type">list</div></td> <td>The final vocabulary.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Unsupervised/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Unsupervised/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Unsupervised/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.preprocessing</span> <span class="k">import</span> <span class="n">CountVectorizer</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.vocabulary&quot;</span><span class="p">)</span>
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
<pre>&lt;CountVectorizer&gt;
</pre>
</div>
</div>

</div>
</div>

</div>