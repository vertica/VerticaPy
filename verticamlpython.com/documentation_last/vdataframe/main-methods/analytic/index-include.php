<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.analytic">vDataFrame.analytic<a class="anchor-link" href="#vDataFrame.analytic">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">analytic</span><span class="p">(</span><span class="n">func</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
                    <span class="n">column</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
                    <span class="n">by</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span> 
                    <span class="n">order_by</span> <span class="o">=</span> <span class="p">[],</span> 
                    <span class="n">column2</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span> 
                    <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">,</span>
                    <span class="n">offset</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                    <span class="n">x_smoothing</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span>
                    <span class="n">add_count</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Adds a new vcolumn to the vDataFrame by using an advanced analytical function on one or two specific vcolumns.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">func</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Function to use.<br>
                                                    <ul>
                                                        <li><b>beta :</b> Beta Coefficient between 2 vcolumns</li>
                                                        <li><b>count :</b> number of non-missing elements</li>
                                                        <li><b>corr :</b> Pearson correlation between 2 vcolumns</li>
                                                        <li><b>cov :</b> covariance between 2 vcolumns</li>
                                                        <li><b>ema :</b> exponential moving average</li>
                                                        <li><b>first_value :</b> first non null lead</li>
                                                        <li><b>iqr :</b> interquartile range</li>
                                                        <li><b>dense_rank :</b> dense rank</li>
                                                        <li><b>kurtosis :</b> kurtosis</li>
                                                        <li><b>jb :</b> Jarque Bera index </li>
                                                        <li><b>lead :</b> next element</li>
                                                        <li><b>lag :</b> previous element</li>
                                                        <li><b>last_value :</b> first non null lag</li>
                                                        <li><b>mad :</b> median absolute deviation</li>
                                                        <li><b>mae :</b> mean absolute error (deviation)</li>
                                                        <li><b>max :</b> maximum</li>
                                                        <li><b>mean :</b> average</li>
                                                        <li><b>median :</b> median</li>
                                                        <li><b>min :</b> min</li>
                                                        <li><b>mode :</b> most occurent element</li>
                                                        <li><b>q% :</b> q quantile (ex: 50% for the median)</li>
                                                        <li><b>pct_change :</b> ratio between the current value and the previous one</li>
                                                        <li><b>percent_rank :</b> percent rank</li>
                                                        <li><b>prod :</b> product</li>
                                                        <li><b>range :</b> difference between the max and the min</li>
                                                        <li><b>rank :</b> rank</li>
                                                        <li><b>row_number :</b> row number</li>
                                                        <li><b>sem :</b> standard error of the mean</li>
                                                        <li><b>skewness :</b> skewness</li>
                                                        <li><b>sum :</b> sum</li>
                                                        <li><b>std :</b> standard deviation</li>
                                                        <li><b>unique :</b> cardinality (count distinct)</li>
                                                        <li><b>var :</b> variance</li></ul>
                                                        Other analytical functions could work if it is part of the DB version you are using.</td> </tr>
    <tr> <td><div class="param_name">column</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Input vcolumn.</td> </tr>
     <tr> <td><div class="param_name">by</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>vcolumns used in the partition.</td> </tr>
    <tr> <td><div class="param_name">order_by</div></td> <td><div class="type">dict / list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the vcolumns used to sort the data using asc order or dictionary of all the sorting methods. For example, to sort by "column1" ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}</td> </tr>
    <tr> <td><div class="param_name">column2</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Second input vcolumn in case of functions using 2 parameters.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the new vcolumn. If empty a default name based on the other parameters will be generated.</td> </tr>
    <tr> <td><div class="param_name">offset</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Lead/Lag offset if parameter 'func' is the function 'lead'/'lag'.</td> </tr>
    <tr> <td><div class="param_name">x_smoothing</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>The smoothing parameter of the 'ema' if the function is 'ema'. It must be in [0;1]</td> </tr>
    <tr> <td><div class="param_name">add_count</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If the function is the 'mode' and this parameter is True then another column will be added to the vDataFrame with the mode number of occurences.</td> </tr>

</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">flights</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.flights&quot;</span><span class="p">)</span>
<span class="n">flights</span><span class="o">.</span><span class="n">eval</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;week&quot;</span><span class="p">,</span> <span class="n">expr</span> <span class="o">=</span> <span class="s2">&quot;WEEK(scheduled_departure)&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">flights</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>week</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">14</td><td style="border: 1px solid white;">DTW</td><td style="border: 1px solid white;">2015-08-16 20:12:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">34</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">29</td><td style="border: 1px solid white;">DTW</td><td style="border: 1px solid white;">2015-08-17 10:07:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">27</td><td style="border: 1px solid white;">34</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">19</td><td style="border: 1px solid white;">ATL</td><td style="border: 1px solid white;">2015-08-17 10:25:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">10</td><td style="border: 1px solid white;">34</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">ORD</td><td style="border: 1px solid white;">2015-08-17 14:00:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">61</td><td style="border: 1px solid white;">34</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">-5</td><td style="border: 1px solid white;">DTW</td><td style="border: 1px solid white;">2015-08-17 14:12:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">-17</td><td style="border: 1px solid white;">34</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 7
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># LAG of departure_delay for the same flight (same airline and same </span>
<span class="c1"># origin/destination airports)</span>
<span class="n">flights</span><span class="o">.</span><span class="n">analytic</span><span class="p">(</span><span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;lag&quot;</span><span class="p">,</span>
                 <span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;departure_delay&quot;</span><span class="p">,</span>
                 <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;origin_airport&quot;</span><span class="p">,</span> <span class="s2">&quot;destination_airport&quot;</span><span class="p">,</span> <span class="s2">&quot;airline&quot;</span><span class="p">],</span>
                 <span class="n">order_by</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;scheduled_departure&quot;</span><span class="p">:</span> <span class="s2">&quot;asc&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>week</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>lag_departure_delay__by_origin_airport_destination_airport_airline_order_by_scheduled_departure</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-14</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">None</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">-2</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-02 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-14</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">-3</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">-2</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-03 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-16</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">-2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-7</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-04 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-13</td><td style="border: 1px solid white;">41</td><td style="border: 1px solid white;">-2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-05 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">41</td><td style="border: 1px solid white;">-7</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 8</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Airline having the biggest number of flights to manage in the week</span>
<span class="n">flights</span><span class="o">.</span><span class="n">analytic</span><span class="p">(</span><span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;mode&quot;</span><span class="p">,</span>
                 <span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;airline&quot;</span><span class="p">,</span>
                 <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;origin_airport&quot;</span><span class="p">,</span> <span class="s2">&quot;week&quot;</span><span class="p">],</span>
                 <span class="n">add_count</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>week</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>mode_airline__by_origin_airport_week</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>mode_airline__by_origin_airport_week_count</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-6</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 12:00:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">10</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">13</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">-5</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 16:00:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">-1</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">13</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-02 12:00:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">13</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-5</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-02 16:00:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">-5</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">13</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">-9</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-03 14:00:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">-14</td><td style="border: 1px solid white;">40</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">13</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 9</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Correlation between the arrival delay and departure delay for the </span>
<span class="c1"># same origin and destination airports</span>
<span class="n">flights</span><span class="o">.</span><span class="n">analytic</span><span class="p">(</span><span class="n">func</span> <span class="o">=</span> <span class="s2">&quot;corr&quot;</span><span class="p">,</span>
                 <span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;departure_delay&quot;</span><span class="p">,</span>
                 <span class="n">column2</span> <span class="o">=</span> <span class="s2">&quot;arrival_delay&quot;</span><span class="p">,</span>
                 <span class="n">by</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;origin_airport&quot;</span><span class="p">,</span> <span class="s2">&quot;destination_airport&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>week</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay_meanby_origin_airport_destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay_meanby_origin_airport_destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>corr_departure_delay_arrival_delay_by_origin_airport_destination_airport</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-31 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-28</td><td style="border: 1px solid white;">44</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">0.933563722644589</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-31 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-13</td><td style="border: 1px solid white;">44</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">0.933563722644589</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-30 21:06:00</td><td style="border: 1px solid white;">DL</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-14</td><td style="border: 1px solid white;">44</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">0.933563722644589</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-30 14:44:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-6</td><td style="border: 1px solid white;">44</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">0.933563722644589</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">29</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-30 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">24</td><td style="border: 1px solid white;">44</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">5.54651162790698</td><td style="border: 1px solid white;">0.933563722644589</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 10</pre>
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
    <tr><td><a href="../eval/index.php">vDataFrame.eval</a></td> <td> Evaluates a customized expression.</td></tr>
    <tr><td><a href="../rolling/index.php">vDataFrame.rolling</a></td> <td> Computes a customized moving window.</td></tr>
</table>
</div>
</div>
</div>