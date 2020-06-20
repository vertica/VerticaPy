<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="regressor.predict">regressor.predict<a class="anchor-link" href="#regressor.predict">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">regressor</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">vdf</span><span class="p">,</span>
                  <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Adds the prediction in a vDataFrame.</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">vdf</div></td> <td><div class="type">vDataFrame</div></td> <td><div class = "no">&#10060;</div></td> <td>Object used to insert the prediction as a vcolumn.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the added vcolumn. If empty, a name will be generated.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : the input object.</p>

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
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">winequality</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.winequality&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">winequality</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>free_sulfur_dioxide</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>density</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>good</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>quality</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>chlorides</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>alcohol</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>color</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pH</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>volatile_acidity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>citric_acid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fixed_acidity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>total_sulfur_dioxide</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sulphates</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>residual_sugar</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">20.00</td><td style="border: 1px solid white;">0.99248</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">0.036</td><td style="border: 1px solid white;">12.4</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.750</td><td style="border: 1px solid white;">0.3100</td><td style="border: 1px solid white;">0.020</td><td style="border: 1px solid white;">3.800</td><td style="border: 1px solid white;">114.00</td><td style="border: 1px solid white;">0.440</td><td style="border: 1px solid white;">11.100</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">29.00</td><td style="border: 1px solid white;">0.989</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">0.03</td><td style="border: 1px solid white;">12.8</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.570</td><td style="border: 1px solid white;">0.2250</td><td style="border: 1px solid white;">0.400</td><td style="border: 1px solid white;">3.900</td><td style="border: 1px solid white;">118.00</td><td style="border: 1px solid white;">0.360</td><td style="border: 1px solid white;">4.200</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">93.00</td><td style="border: 1px solid white;">0.98999</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7</td><td style="border: 1px solid white;">0.029</td><td style="border: 1px solid white;">12.0</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.650</td><td style="border: 1px solid white;">0.1700</td><td style="border: 1px solid white;">0.360</td><td style="border: 1px solid white;">4.200</td><td style="border: 1px solid white;">161.00</td><td style="border: 1px solid white;">0.890</td><td style="border: 1px solid white;">1.800</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">64.00</td><td style="border: 1px solid white;">0.99688</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0.041</td><td style="border: 1px solid white;">8.0</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.420</td><td style="border: 1px solid white;">0.2150</td><td style="border: 1px solid white;">0.230</td><td style="border: 1px solid white;">4.200</td><td style="border: 1px solid white;">157.00</td><td style="border: 1px solid white;">0.440</td><td style="border: 1px solid white;">5.100</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">31.00</td><td style="border: 1px solid white;">0.98904</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">0.03</td><td style="border: 1px solid white;">12.8</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.460</td><td style="border: 1px solid white;">0.3200</td><td style="border: 1px solid white;">0.390</td><td style="border: 1px solid white;">4.400</td><td style="border: 1px solid white;">127.00</td><td style="border: 1px solid white;">0.360</td><td style="border: 1px solid white;">4.300</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: winequality, Number of rows: 6497, Number of columns: 14
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestRegressor</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestRegressor</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.RF_winequality&quot;</span><span class="p">,</span>
                              <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                              <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                              <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                              <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                              <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                              <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                              <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                              <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.winequality&quot;</span><span class="p">,</span> 
          <span class="p">[</span><span class="s2">&quot;alcohol&quot;</span><span class="p">,</span> <span class="s2">&quot;fixed_acidity&quot;</span><span class="p">,</span> <span class="s2">&quot;pH&quot;</span><span class="p">,</span> <span class="s2">&quot;density&quot;</span><span class="p">],</span> 
          <span class="s2">&quot;quality&quot;</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">winequality</span><span class="p">,</span> <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;predict_quality&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>free_sulfur_dioxide</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>density</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>good</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>quality</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>chlorides</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>alcohol</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>color</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>pH</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>volatile_acidity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>citric_acid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fixed_acidity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>total_sulfur_dioxide</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sulphates</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>residual_sugar</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>RandomForestRegressor_publicRFwinequality</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>predict_quality</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">20.00</td><td style="border: 1px solid white;">0.99248</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">0.036</td><td style="border: 1px solid white;">12.4</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.750</td><td style="border: 1px solid white;">0.3100</td><td style="border: 1px solid white;">0.020</td><td style="border: 1px solid white;">3.800</td><td style="border: 1px solid white;">114.00</td><td style="border: 1px solid white;">0.440</td><td style="border: 1px solid white;">11.100</td><td style="border: 1px solid white;">6.3787333212411</td><td style="border: 1px solid white;">6.3787333212411</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">29.00</td><td style="border: 1px solid white;">0.989</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">0.03</td><td style="border: 1px solid white;">12.8</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.570</td><td style="border: 1px solid white;">0.2250</td><td style="border: 1px solid white;">0.400</td><td style="border: 1px solid white;">3.900</td><td style="border: 1px solid white;">118.00</td><td style="border: 1px solid white;">0.360</td><td style="border: 1px solid white;">4.200</td><td style="border: 1px solid white;">6.58626788584711</td><td style="border: 1px solid white;">6.58626788584711</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">93.00</td><td style="border: 1px solid white;">0.98999</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">7</td><td style="border: 1px solid white;">0.029</td><td style="border: 1px solid white;">12.0</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.650</td><td style="border: 1px solid white;">0.1700</td><td style="border: 1px solid white;">0.360</td><td style="border: 1px solid white;">4.200</td><td style="border: 1px solid white;">161.00</td><td style="border: 1px solid white;">0.890</td><td style="border: 1px solid white;">1.800</td><td style="border: 1px solid white;">6.44583178840808</td><td style="border: 1px solid white;">6.44583178840808</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">64.00</td><td style="border: 1px solid white;">0.99688</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">0.041</td><td style="border: 1px solid white;">8.0</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.420</td><td style="border: 1px solid white;">0.2150</td><td style="border: 1px solid white;">0.230</td><td style="border: 1px solid white;">4.200</td><td style="border: 1px solid white;">157.00</td><td style="border: 1px solid white;">0.440</td><td style="border: 1px solid white;">5.100</td><td style="border: 1px solid white;">5.49278049271711</td><td style="border: 1px solid white;">5.49278049271711</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">31.00</td><td style="border: 1px solid white;">0.98904</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8</td><td style="border: 1px solid white;">0.03</td><td style="border: 1px solid white;">12.8</td><td style="border: 1px solid white;">white</td><td style="border: 1px solid white;">3.460</td><td style="border: 1px solid white;">0.3200</td><td style="border: 1px solid white;">0.390</td><td style="border: 1px solid white;">4.400</td><td style="border: 1px solid white;">127.00</td><td style="border: 1px solid white;">0.360</td><td style="border: 1px solid white;">4.300</td><td style="border: 1px solid white;">6.58626788584711</td><td style="border: 1px solid white;">6.58626788584711</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[41]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: winequality, Number of rows: 6497, Number of columns: 16</pre>
</div>

</div>

</div>
</div>

</div>