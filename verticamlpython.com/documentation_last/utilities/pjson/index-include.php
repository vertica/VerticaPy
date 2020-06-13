<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="pjson">pjson<a class="anchor-link" href="#pjson">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">pjson</span><span class="p">(</span><span class="n">path</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
      <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Parses a JSON file using flex tables. It will identify the columns and their respective types.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">path</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Absolute path where the JSON file is located.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
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
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># Parsing the JSON file</span>
<span class="n">pjson</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[1]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;datasetid&#39;: &#39;Varchar(36)&#39;,
 &#39;fields.age&#39;: &#39;Float&#39;,
 &#39;fields.cabin&#39;: &#39;Varchar(30)&#39;,
 &#39;fields.embarked&#39;: &#39;Varchar(20)&#39;,
 &#39;fields.fare&#39;: &#39;Float&#39;,
 &#39;fields.name&#39;: &#39;Varchar(164)&#39;,
 &#39;fields.parch&#39;: &#39;Integer&#39;,
 &#39;fields.passengerid&#39;: &#39;Integer&#39;,
 &#39;fields.pclass&#39;: &#39;Integer&#39;,
 &#39;fields.sex&#39;: &#39;Varchar(20)&#39;,
 &#39;fields.sibsp&#39;: &#39;Integer&#39;,
 &#39;fields.survived&#39;: &#39;Boolean&#39;,
 &#39;fields.ticket&#39;: &#39;Varchar(36)&#39;,
 &#39;record_timestamp&#39;: &#39;Timestamp&#39;,
 &#39;recordid&#39;: &#39;Uuid&#39;}</pre>
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