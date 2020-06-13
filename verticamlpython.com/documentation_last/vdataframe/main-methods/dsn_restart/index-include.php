<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.dsn_restart">vDataFrame.dsn_restart<a class="anchor-link" href="#vDataFrame.dsn_restart">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">dsn_restart</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a new cursor based on the DSN used to build the vDataFrame.</p>

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
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">titanic</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.titanic&quot;</span><span class="p">,</span> <span class="n">dsn</span> <span class="o">=</span> <span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>

<span class="c1"># If the connection was lost, you can re-create a new cursor using dsn_restart</span>
<span class="c1"># It is possible if the vDataFrame was created using a DSN or if you saved a </span>
<span class="c1"># DSN using the method &#39;set_dsn&#39;</span>
<span class="n">titanic</span><span class="o">.</span><span class="n">dsn_restart</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="See-Also">See Also<a class="anchor-link" href="#See-Also">&#182;</a></h3><table id="seealso">
    <tr><td><a href="../set_dsn">vDataFrame.set_dsn</a></td> <td>Sets a new DB DSN.</td></tr>
</table>
</div>
</div>
</div>