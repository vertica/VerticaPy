<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="pcsv">pcsv<a class="anchor-link" href="#pcsv">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">pcsv</span><span class="p">(</span><span class="n">path</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
     <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
     <span class="n">sep</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;,&#39;</span><span class="p">,</span>
     <span class="n">header</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
     <span class="n">header_names</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
     <span class="n">na_rep</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span> 
     <span class="n">quotechar</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&quot;&#39;</span><span class="p">,</span>
     <span class="n">escape</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="se">\\</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Parses a CSV file using flex tables. It will identify the columns and their respective types.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">path</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Absolute path where the CSV file is located.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">sep</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Column separator.</td> </tr>
    <tr> <td><div class="param_name">header</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to False, the parameter 'header_names' will be used to name the different columns.</td> </tr>
    <tr> <td><div class="param_name">header_names</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the columns names.</td> </tr>
    <tr> <td><div class="param_name">na_rep</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Missing values representation.</td> </tr>
    <tr> <td><div class="param_name">quotechar</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Char which is enclosing the str values.</td> </tr>
    <tr> <td><div class="param_name">escape</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Separator between each record.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>dict</b> : dictionary containing for each column its type.</p>

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
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># Parses the CSV file </span>
<span class="n">pcsv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span> 
     <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
     <span class="n">na_rep</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[32]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;age&#39;: &#39;Numeric(6,3)&#39;,
 &#39;boat&#39;: &#39;Varchar(100)&#39;,
 &#39;body&#39;: &#39;Integer&#39;,
 &#39;cabin&#39;: &#39;Varchar(30)&#39;,
 &#39;embarked&#39;: &#39;Varchar(20)&#39;,
 &#39;fare&#39;: &#39;Numeric(10,5)&#39;,
 &#39;home.dest&#39;: &#39;Varchar(100)&#39;,
 &#39;name&#39;: &#39;Varchar(164)&#39;,
 &#39;parch&#39;: &#39;Integer&#39;,
 &#39;pclass&#39;: &#39;Integer&#39;,
 &#39;sex&#39;: &#39;Varchar(20)&#39;,
 &#39;sibsp&#39;: &#39;Integer&#39;,
 &#39;survived&#39;: &#39;Integer&#39;,
 &#39;ticket&#39;: &#39;Varchar(36)&#39;}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># You can also rename the columns or name them if it has </span>
<span class="c1"># no header by using the parameter &#39;header_names&#39;</span>
<span class="n">pcsv</span><span class="p">(</span><span class="s2">&quot;titanic.csv&quot;</span><span class="p">,</span> 
     <span class="n">sep</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span>
     <span class="n">na_rep</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
     <span class="n">header</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
     <span class="n">header_names</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;new_name1&quot;</span><span class="p">,</span> <span class="s2">&quot;new_name2&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[35]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;new_name1&#39;: &#39;Integer&#39;,
 &#39;new_name2&#39;: &#39;Integer&#39;,
 &#39;ucol10&#39;: &#39;Varchar(20)&#39;,
 &#39;ucol11&#39;: &#39;Varchar(100)&#39;,
 &#39;ucol12&#39;: &#39;Integer&#39;,
 &#39;ucol13&#39;: &#39;Varchar(100)&#39;,
 &#39;ucol2&#39;: &#39;Varchar(164)&#39;,
 &#39;ucol3&#39;: &#39;Varchar(20)&#39;,
 &#39;ucol4&#39;: &#39;Numeric(6,3)&#39;,
 &#39;ucol5&#39;: &#39;Integer&#39;,
 &#39;ucol6&#39;: &#39;Integer&#39;,
 &#39;ucol7&#39;: &#39;Varchar(36)&#39;,
 &#39;ucol8&#39;: &#39;Numeric(10,5)&#39;,
 &#39;ucol9&#39;: &#39;Varchar(30)&#39;}</pre>
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
    <tr><td><a href="../read_csv/index.php">read_csv</a></td> <td>Ingests a CSV file in the Vertica DB.</td></tr>
    <tr><td><a href="../read_json/index.php">read_json</a></td> <td>Ingests a JSON file in the Vertica DB.</td></tr>
</table>
</div>
</div>
</div>