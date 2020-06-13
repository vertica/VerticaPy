<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame[].fill_outliers">vDataFrame[].fill_outliers<a class="anchor-link" href="#vDataFrame[].fill_outliers">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="p">[]</span><span class="o">.</span><span class="n">fill_outliers</span><span class="p">(</span><span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;winsorize&quot;</span><span class="p">,</span>
                           <span class="n">threshold</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">4.0</span><span class="p">,</span> 
                           <span class="n">use_threshold</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
                           <span class="n">alpha</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.05</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Fills the vcolumns outliers using the input method.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Method used to fill the vcolumn outliers.<br>
                                                    <ul>
                                                        <li><b>mean :</b> Replaces the upper and lower outliers by their respective average. </li>
                                                        <li><b>null :</b> Replaces the outliers by the NULL value.</li>
                                                        <li><b>winsorize :</b> Clips the vcolumn using as lower bound quantile(alpha) and as upper bound quantile(1-alpha) if 'use_threshold' is set to False else the lower and upper ZScores.</li>
                                                        </ul></td> </tr>
    <tr> <td><div class="param_name">threshold</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Uses the Gaussian distribution to define the outliers. After normalizing the data (Z-Score), if the absolute value of the record is greater than the threshold it will be considered as an outlier.</td> </tr>
    <tr> <td><div class="param_name">use_threshold</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>Uses the threshold instead of the 'alpha' parameter.</td> </tr>
    <tr> <td><div class="param_name">alpha</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number representing the outliers threshold. Values lesser than quantile(alpha) or greater than quantile(1-alpha) will be filled.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self.parent</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[99]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.datasets</span> <span class="k">import</span> <span class="n">load_market</span>
<span class="n">market</span> <span class="o">=</span> <span class="n">load_market</span><span class="p">()</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="s2">&quot;Price &lt; 0.7&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">market</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">20</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>294 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Form</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Price</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Name</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.5104657455</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.537867915537</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Ready to drink</td><td style="border: 1px solid white;">0.6311325278</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.5494172928</td><td style="border: 1px solid white;">Bananas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.566983414531</td><td style="border: 1px solid white;">Bananas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>5</b></td><td style="border: 1px solid white;">Fresh green cabbage</td><td style="border: 1px solid white;">0.579208394258</td><td style="border: 1px solid white;">Cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>6</b></td><td style="border: 1px solid white;">Fresh green cabbage</td><td style="border: 1px solid white;">0.6238712291</td><td style="border: 1px solid white;">Cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>7</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.520793672</td><td style="border: 1px solid white;">Cantaloupe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>8</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.535873776106</td><td style="border: 1px solid white;">Cantaloupe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>9</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.674839678618</td><td style="border: 1px solid white;">Grapefruit</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>10</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.658986796</td><td style="border: 1px solid white;">Oranges</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>11</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.689934119435</td><td style="border: 1px solid white;">Oranges</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>12</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.627661945936</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>13</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.6527945474</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>14</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.590752357725</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>15</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.6203795725</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>16</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.564319813929</td><td style="border: 1px solid white;">Potatoes</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>17</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.6037360906</td><td style="border: 1px solid white;">Potatoes</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>18</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.3166387792</td><td style="border: 1px solid white;">Watermelon</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>19</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.333412035323</td><td style="border: 1px solid white;">Watermelon</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: market, Number of rows: 20, Number of columns: 3
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[94]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># All the outliers (abs(ZSCORE) &gt; 1.5) will be replaced by the NULL values</span>
<span class="n">market</span><span class="p">[</span><span class="s2">&quot;Price&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fill_outliers</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;null&quot;</span><span class="p">,</span>
                              <span class="n">threshold</span> <span class="o">=</span> <span class="mf">1.5</span><span class="p">,</span>
                              <span class="n">use_threshold</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Form</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Price</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Name</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.537867915537</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Ready to drink</td><td style="border: 1px solid white;">0.6311325278</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.5494172928</td><td style="border: 1px solid white;">Bananas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.566983414531</td><td style="border: 1px solid white;">Bananas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>5</b></td><td style="border: 1px solid white;">Fresh green cabbage</td><td style="border: 1px solid white;">0.579208394258</td><td style="border: 1px solid white;">Cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>6</b></td><td style="border: 1px solid white;">Fresh green cabbage</td><td style="border: 1px solid white;">0.6238712291</td><td style="border: 1px solid white;">Cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>7</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.520793672</td><td style="border: 1px solid white;">Cantaloupe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>8</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.535873776106</td><td style="border: 1px solid white;">Cantaloupe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>9</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.674839678618</td><td style="border: 1px solid white;">Grapefruit</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>10</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.658986796</td><td style="border: 1px solid white;">Oranges</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>11</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Oranges</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>12</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.627661945936</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>13</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.6527945474</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>14</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.590752357725</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>15</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.6203795725</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>16</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.564319813929</td><td style="border: 1px solid white;">Potatoes</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>17</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.6037360906</td><td style="border: 1px solid white;">Potatoes</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>18</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Watermelon</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>19</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">Watermelon</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[94]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: market, Number of rows: 20, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[96]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># All the outliers (abs(ZSCORE) &gt; 1.5) will be replaced by the lower and </span>
<span class="c1"># upper bound having a ZSCORE = 1.5 and -1.5</span>
<span class="n">market</span><span class="p">[</span><span class="s2">&quot;Price&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fill_outliers</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;winsorize&quot;</span><span class="p">,</span>
                              <span class="n">threshold</span> <span class="o">=</span> <span class="mf">1.5</span><span class="p">,</span>
                              <span class="n">use_threshold</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Price</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Name</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Form</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">0.5104657455</td><td style="border: 1px solid white;">Apples</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">0.537867915537</td><td style="border: 1px solid white;">Apples</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">0.6311325278</td><td style="border: 1px solid white;">Apples</td><td style="border: 1px solid white;">Ready to drink</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">0.5494172928</td><td style="border: 1px solid white;">Bananas</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0.566983414531</td><td style="border: 1px solid white;">Bananas</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>5</b></td><td style="border: 1px solid white;">0.579208394258</td><td style="border: 1px solid white;">Cabbage</td><td style="border: 1px solid white;">Fresh green cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>6</b></td><td style="border: 1px solid white;">0.6238712291</td><td style="border: 1px solid white;">Cabbage</td><td style="border: 1px solid white;">Fresh green cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>7</b></td><td style="border: 1px solid white;">0.520793672</td><td style="border: 1px solid white;">Cantaloupe</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>8</b></td><td style="border: 1px solid white;">0.535873776106</td><td style="border: 1px solid white;">Cantaloupe</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>9</b></td><td style="border: 1px solid white;">0.674839678618</td><td style="border: 1px solid white;">Grapefruit</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>10</b></td><td style="border: 1px solid white;">0.658986796</td><td style="border: 1px solid white;">Oranges</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>11</b></td><td style="border: 1px solid white;">0.689934119435</td><td style="border: 1px solid white;">Oranges</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>12</b></td><td style="border: 1px solid white;">0.627661945936</td><td style="border: 1px solid white;">Pineapple</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>13</b></td><td style="border: 1px solid white;">0.6527945474</td><td style="border: 1px solid white;">Pineapple</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>14</b></td><td style="border: 1px solid white;">0.590752357725</td><td style="border: 1px solid white;">Pineapple</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>15</b></td><td style="border: 1px solid white;">0.6203795725</td><td style="border: 1px solid white;">Pineapple</td><td style="border: 1px solid white;">Frozen</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>16</b></td><td style="border: 1px solid white;">0.564319813929</td><td style="border: 1px solid white;">Potatoes</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>17</b></td><td style="border: 1px solid white;">0.6037360906</td><td style="border: 1px solid white;">Potatoes</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>18</b></td><td style="border: 1px solid white;">0.422205378756987</td><td style="border: 1px solid white;">Watermelon</td><td style="border: 1px solid white;">Fresh</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>19</b></td><td style="border: 1px solid white;">0.422205378756987</td><td style="border: 1px solid white;">Watermelon</td><td style="border: 1px solid white;">Fresh</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[96]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: market, Number of rows: 20, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[100]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># All the outliers (values &gt; quantile(0.8) or &lt; quantile(0.8)) will be </span>
<span class="c1"># replaced by the nearest of the two quantiles</span>
<span class="n">market</span><span class="p">[</span><span class="s2">&quot;Price&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fill_outliers</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;winsorize&quot;</span><span class="p">,</span>
                              <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.2</span><span class="p">,</span>
                              <span class="n">use_threshold</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Form</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Price</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>Name</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.5328577552848</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.537867915537</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Ready to drink</td><td style="border: 1px solid white;">0.6311325278</td><td style="border: 1px solid white;">Apples</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.5494172928</td><td style="border: 1px solid white;">Bananas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.566983414531</td><td style="border: 1px solid white;">Bananas</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>5</b></td><td style="border: 1px solid white;">Fresh green cabbage</td><td style="border: 1px solid white;">0.579208394258</td><td style="border: 1px solid white;">Cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>6</b></td><td style="border: 1px solid white;">Fresh green cabbage</td><td style="border: 1px solid white;">0.6238712291</td><td style="border: 1px solid white;">Cabbage</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>7</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.5328577552848</td><td style="border: 1px solid white;">Cantaloupe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>8</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.535873776106</td><td style="border: 1px solid white;">Cantaloupe</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>9</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.63546493172</td><td style="border: 1px solid white;">Grapefruit</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>10</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.63546493172</td><td style="border: 1px solid white;">Oranges</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>11</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.63546493172</td><td style="border: 1px solid white;">Oranges</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>12</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.627661945936</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>13</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.63546493172</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>14</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.590752357725</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>15</b></td><td style="border: 1px solid white;">Frozen</td><td style="border: 1px solid white;">0.6203795725</td><td style="border: 1px solid white;">Pineapple</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>16</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.564319813929</td><td style="border: 1px solid white;">Potatoes</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>17</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.6037360906</td><td style="border: 1px solid white;">Potatoes</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>18</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.5328577552848</td><td style="border: 1px solid white;">Watermelon</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>19</b></td><td style="border: 1px solid white;">Fresh</td><td style="border: 1px solid white;">0.5328577552848</td><td style="border: 1px solid white;">Watermelon</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[100]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: market, Number of rows: 20, Number of columns: 3</pre>
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
    <tr><td><a href="../drop_outliers">vDataFrame[].drop_outliers</a></td> <td>Drops the vcolumn outliers.</td></tr>
    <tr><td><a href="../../main-methods/outliers">vDataFrame.outliers</a></td> <td>Computes the vDataFrame Global Outliers.</td></tr>
</table>
</div>
</div>
</div>