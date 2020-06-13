<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="classifier.deploySQL">classifier.deploySQL<a class="anchor-link" href="#classifier.deploySQL">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span><span class="o">.</span><span class="n">deploySQL</span><span class="p">(</span><span class="n">predict</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Returns the SQL code needed to deploy the model.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">predict</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, returns the prediction instead of the probability.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>str / list</b> : the SQL code needed to deploy the model.</p>

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
<div class="prompt input_prompt">In&nbsp;[51]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.neighbors</span> <span class="k">import</span> <span class="n">KNeighborsClassifier</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">KNeighborsClassifier</span><span class="p">()</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> 
          <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalWidthCm&quot;</span><span class="p">],</span> 
          <span class="s2">&quot;Species&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">deploySQL</span><span class="p">())</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>(SELECT row_id, &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34;, &#34;Species&#34;, predict_knc, COUNT(*) / 5 AS proba_predict FROM (SELECT x.&#34;PetalLengthCm&#34;, x.&#34;PetalWidthCm&#34;, x.&#34;SepalWidthCm&#34;, ROW_NUMBER() OVER(PARTITION BY x.&#34;PetalLengthCm&#34;, x.&#34;PetalWidthCm&#34;, x.&#34;SepalWidthCm&#34;, row_id ORDER BY POWER(POWER(ABS(x.&#34;PetalLengthCm&#34; - y.&#34;PetalLengthCm&#34;), 2) + POWER(ABS(x.&#34;PetalWidthCm&#34; - y.&#34;PetalWidthCm&#34;), 2) + POWER(ABS(x.&#34;SepalWidthCm&#34; - y.&#34;SepalWidthCm&#34;), 2), 1 / 2)) AS ordered_distance, x.&#34;Species&#34;, y.&#34;Species&#34; AS predict_knc, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM public.iris WHERE &#34;PetalLengthCm&#34; IS NOT NULL AND &#34;PetalWidthCm&#34; IS NOT NULL AND &#34;SepalWidthCm&#34; IS NOT NULL) x CROSS JOIN (SELECT * FROM public.iris WHERE &#34;PetalLengthCm&#34; IS NOT NULL AND &#34;PetalWidthCm&#34; IS NOT NULL AND &#34;SepalWidthCm&#34; IS NOT NULL) y) z WHERE ordered_distance &lt;= 5 GROUP BY &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34;, &#34;Species&#34;, row_id, predict_knc) knc_table
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[52]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">deploySQL</span><span class="p">(</span><span class="n">predict</span> <span class="o">=</span> <span class="kc">True</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>(SELECT &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34;, &#34;Species&#34;, predict_knc FROM (SELECT &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34;, &#34;Species&#34;, predict_knc, ROW_NUMBER() OVER (PARTITION BY &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34; ORDER BY proba_predict DESC) AS order_prediction FROM (SELECT row_id, &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34;, &#34;Species&#34;, predict_knc, COUNT(*) / 5 AS proba_predict FROM (SELECT x.&#34;PetalLengthCm&#34;, x.&#34;PetalWidthCm&#34;, x.&#34;SepalWidthCm&#34;, ROW_NUMBER() OVER(PARTITION BY x.&#34;PetalLengthCm&#34;, x.&#34;PetalWidthCm&#34;, x.&#34;SepalWidthCm&#34;, row_id ORDER BY POWER(POWER(ABS(x.&#34;PetalLengthCm&#34; - y.&#34;PetalLengthCm&#34;), 2) + POWER(ABS(x.&#34;PetalWidthCm&#34; - y.&#34;PetalWidthCm&#34;), 2) + POWER(ABS(x.&#34;SepalWidthCm&#34; - y.&#34;SepalWidthCm&#34;), 2), 1 / 2)) AS ordered_distance, x.&#34;Species&#34;, y.&#34;Species&#34; AS predict_knc, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM public.iris WHERE &#34;PetalLengthCm&#34; IS NOT NULL AND &#34;PetalWidthCm&#34; IS NOT NULL AND &#34;SepalWidthCm&#34; IS NOT NULL) x CROSS JOIN (SELECT * FROM public.iris WHERE &#34;PetalLengthCm&#34; IS NOT NULL AND &#34;PetalWidthCm&#34; IS NOT NULL AND &#34;SepalWidthCm&#34; IS NOT NULL) y) z WHERE ordered_distance &lt;= 5 GROUP BY &#34;PetalLengthCm&#34;, &#34;PetalWidthCm&#34;, &#34;SepalWidthCm&#34;, &#34;Species&#34;, row_id, predict_knc) knc_table) x WHERE order_prediction = 1) predict_knc_table
</pre>
</div>
</div>

</div>
</div>

</div>