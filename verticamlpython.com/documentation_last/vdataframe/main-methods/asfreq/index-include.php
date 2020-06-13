<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.asfreq">vDataFrame.asfreq<a class="anchor-link" href="#vDataFrame.asfreq">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">asfreq</span><span class="p">(</span><span class="n">ts</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                  <span class="n">rule</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                  <span class="n">method</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span>
                  <span class="n">by</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes a regular time interval vDataFrame by interpolating the missing values using different techniques.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">ts</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>TS (Time Series) vcolumn used to order the data. The vcolumn type must be date like (date, datetime, timestamp...)</td> </tr>
    <tr> <td><div class="param_name">rule</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Interval used to slice the time. For example, '5 minutes' will create records separated by '5 minutes' time interval.</td> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">dict</div></td> <td><div class = "no">&#10060;</div></td> <td>Dictionary of all the different methods of interpolation. The dict must be similar to the following: {"column1": "interpolation1" ..., "columnk": "interpolationk"}. 3 types of interpolations are possible:<br>
                                                    <ul>
                                                        <li><b>bfill :</b> Constant propagation of the next value (Back Propagation).</li>
                                                        <li><b>ffill :</b> Constant propagation of the first value (First Propagation).</li>
                                                        <li><b>linear :</b> Linear Interpolation.</li></ul></td> </tr>
    <tr> <td><div class="param_name">by</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vcolumns used in the partition.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : object result of the interpolation.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_smart_meters</span>
<span class="n">sm</span> <span class="o">=</span> <span class="n">load_smart_meters</span><span class="p">()</span>
<span class="nb">print</span><span class="p">(</span><span class="n">sm</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>id</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>val</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>time</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">0.0370000</td><td style="border: 1px solid white;">2014-01-01 01:15:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">0.0800000</td><td style="border: 1px solid white;">2014-01-01 02:30:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0.0810000</td><td style="border: 1px solid white;">2014-01-01 03:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">1.4890000</td><td style="border: 1px solid white;">2014-01-01 05:00:00</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">0.0720000</td><td style="border: 1px solid white;">2014-01-01 06:00:00</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: smart_meters, Number of rows: 11844, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Linear interpolation by hour of each smart meter value</span>
<span class="n">sm</span><span class="o">.</span><span class="n">asfreq</span><span class="p">(</span><span class="n">ts</span> <span class="o">=</span> <span class="s2">&quot;time&quot;</span><span class="p">,</span>
          <span class="n">rule</span> <span class="o">=</span> <span class="s2">&quot;1 hour&quot;</span><span class="p">,</span>
          <span class="n">method</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;val&quot;</span><span class="p">:</span> <span class="s2">&quot;linear&quot;</span><span class="p">},</span>
          <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;id&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>time</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>id</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>val</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">2014-01-01 11:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.029</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">2014-01-01 12:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.119181818181818</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">2014-01-01 13:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.209363636363636</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">2014-01-01 14:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.27752380952381</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">2014-01-01 15:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.279619047619048</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>5</b></td><td style="border: 1px solid white;">2014-01-01 16:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.281714285714286</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>6</b></td><td style="border: 1px solid white;">2014-01-01 17:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.283809523809524</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>7</b></td><td style="border: 1px solid white;">2014-01-01 18:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.285904761904762</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>8</b></td><td style="border: 1px solid white;">2014-01-01 19:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.288</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>9</b></td><td style="border: 1px solid white;">2014-01-01 20:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.290095238095238</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>10</b></td><td style="border: 1px solid white;">2014-01-01 21:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.292190476190476</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>11</b></td><td style="border: 1px solid white;">2014-01-01 22:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.294285714285714</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>12</b></td><td style="border: 1px solid white;">2014-01-01 23:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.296380952380952</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>13</b></td><td style="border: 1px solid white;">2014-01-02 00:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.298476190476191</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>14</b></td><td style="border: 1px solid white;">2014-01-02 01:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.300571428571429</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>15</b></td><td style="border: 1px solid white;">2014-01-02 02:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.302666666666667</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>16</b></td><td style="border: 1px solid white;">2014-01-02 03:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.304761904761905</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>17</b></td><td style="border: 1px solid white;">2014-01-02 04:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.306857142857143</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>18</b></td><td style="border: 1px solid white;">2014-01-02 05:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.308952380952381</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>19</b></td><td style="border: 1px solid white;">2014-01-02 06:00:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.311047619047619</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[19]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: asfreq, Number of rows: 148189, Number of columns: 3</pre>
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
    <tr><td><a href="../../vcolumn-methods/fillna">vDataFrame[].fillna</a></td> <td> Fills the vcolumn missing values.</td></tr>
    <tr><td><a href="../../vcolumn-methods/slice">vDataFrame[].slice</a></td> <td> Slices the vcolumn.</td></tr>
</table>
</div>
</div>
</div>