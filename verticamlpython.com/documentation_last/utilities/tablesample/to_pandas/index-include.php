<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="tablesample.to_pandas">tablesample.to_pandas<a class="anchor-link" href="#tablesample.to_pandas">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">tablesample</span><span class="o">.</span><span class="n">to_pandas</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Converts the tablesample to a pandas DataFrame.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>pandas.DataFrame</b> : pandas DataFrame of the tablesample.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">dataset</span> <span class="o">=</span> <span class="n">tablesample</span><span class="p">(</span><span class="n">values</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;index&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;region&quot;</span><span class="p">,</span> <span class="s2">&quot;price&quot;</span><span class="p">,</span> <span class="s2">&quot;quality&quot;</span><span class="p">],</span>
                                <span class="s2">&quot;Banana&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Brazil&quot;</span><span class="p">,</span> <span class="mf">2.3</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span>
                                <span class="s2">&quot;Manguo&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;Columbia&quot;</span><span class="p">,</span> <span class="mf">6.7</span><span class="p">,</span> <span class="mi">5</span><span class="p">],</span>
                                <span class="s2">&quot;Apple&quot;</span><span class="p">:</span> <span class="p">[</span><span class="s2">&quot;France&quot;</span><span class="p">,</span> <span class="mf">1.5</span><span class="p">,</span> <span class="mi">2</span><span class="p">]},</span>
                      <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;Fruits&quot;</span><span class="p">)</span>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Banana</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Manguo</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Apple</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>region</b></td><td style="border: 1px solid white;">Brazil</td><td style="border: 1px solid white;">Columbia</td><td style="border: 1px solid white;">France</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>price</b></td><td style="border: 1px solid white;">2.3</td><td style="border: 1px solid white;">6.7</td><td style="border: 1px solid white;">1.5</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>quality</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">2</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: Fruits, Number of rows: 3, Number of columns: 4
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dataset</span><span class="o">.</span><span class="n">to_pandas</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[17]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Banana</th>
      <th>Manguo</th>
      <th>Apple</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>region</th>
      <td>Brazil</td>
      <td>Columbia</td>
      <td>France</td>
    </tr>
    <tr>
      <th>price</th>
      <td>2.3</td>
      <td>6.7</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>