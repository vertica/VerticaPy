<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Descriptive-Statistics">Descriptive Statistics<a class="anchor-link" href="#Descriptive-Statistics">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The easiest way to understand the data is to aggregate. 
An aggregation is a number or a category which summarizes the data. Vertica ML Python allows the computation of all the well known aggregation in one line.</p>
<p>Using the 'agg' method is the best way to compute multiple aggregations on multiple columns at the same time.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">help</span><span class="p">(</span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">agg</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function agg in module vertica_ml_python.vdataframe:

agg(self, func:list, columns:list=[])
    ---------------------------------------------------------------------------
    Aggregates the vDataFrame using the input functions.
    
    Parameters
    ----------
    func: list
            List of the different aggregation.
                    approx_unique  : approximative cardinality
                    count          : number of non-missing elements
                    cvar           : conditional value at risk
                    dtype          : virtual column type
                    iqr            : interquartile range
                    kurtosis       : kurtosis
                    jb             : Jarque Bera index 
                    mad            : median absolute deviation
                    mae            : mean absolute error (deviation)
                    max            : maximum
                    mean           : average
                    median         : median
                    min            : minimum
                    mode           : most occurent element
                    percent        : percent of non-missing elements
                    q%             : q quantile (ex: 50% for the median)
                    prod           : product
                    range          : difference between the max and the min
                    sem            : standard error of the mean
                    skewness       : skewness
                    sum            : sum
                    std            : standard deviation
                    topk           : kth most occurent element (ex: top1 for the mode)
                    topk_percent   : kth most occurent element density
                    unique         : cardinality (count distinct)
                    var            : variance
                            Other aggregations could work if it is part of 
                            the DB version you are using.
    columns: list, optional
            List of the vcolumns names. If empty, all the vcolumns 
            or only numerical vcolumns will be used depending on the
            aggregations.
    
    Returns
    -------
    tablesample
            An object containing the result. For more information, check out
            utilities.tablesample.
    
    See Also
    --------
    vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
            analytical function on a specific vcolumn.

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This function will help you understanding your data.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.churn&quot;</span><span class="p">)</span>
<span class="n">vdf</span><span class="o">.</span><span class="n">agg</span><span class="p">(</span><span class="n">func</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;min&quot;</span><span class="p">,</span> <span class="s2">&quot;10%&quot;</span><span class="p">,</span> <span class="s2">&quot;median&quot;</span><span class="p">,</span> <span class="s2">&quot;90%&quot;</span><span class="p">,</span> <span class="s2">&quot;max&quot;</span><span class="p">,</span> <span class="s2">&quot;kurtosis&quot;</span><span class="p">,</span> <span class="s2">&quot;unique&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>min</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>10%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>median</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>90%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>max</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>kurtosis</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Churn"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">-0.870211342331981</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"TotalCharges"</b></td><td style="border: 1px solid white;">18.8</td><td style="border: 1px solid white;">84.445</td><td style="border: 1px solid white;">1397.475</td><td style="border: 1px solid white;">5985.4476923077</td><td style="border: 1px solid white;">8684.8</td><td style="border: 1px solid white;">-0.231798760869362</td><td style="border: 1px solid white;">6530</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"tenure"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">69.0</td><td style="border: 1px solid white;">72</td><td style="border: 1px solid white;">-1.38737163597169</td><td style="border: 1px solid white;">73</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Dependents"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">-1.2343780571695</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"MonthlyCharges"</b></td><td style="border: 1px solid white;">18.25</td><td style="border: 1px solid white;">20.05</td><td style="border: 1px solid white;">70.3214285714286</td><td style="border: 1px solid white;">102.6</td><td style="border: 1px solid white;">118.75</td><td style="border: 1px solid white;">-1.25725969454951</td><td style="border: 1px solid white;">1585</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PhoneService"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">5.43890755508706</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PaperlessBilling"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">-1.85960618560884</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"SeniorCitizen"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1.36259589579391</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Partner"</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">-1.9959534211947</td><td style="border: 1px solid white;">2</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[36]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Some other methods are abstractions of the 'agg' method. They will simplify the call to specific aggregations computations. You can use the 'statistics' method to get in one line the most useful quantiles and other important statistics.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">statistics</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>skewness</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>kurtosis</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>stddev</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>min</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>10%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>25%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>median</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>75%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>90%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>max</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Churn"</b></td><td style="border: 1px solid white;">1.06303144457513</td><td style="border: 1px solid white;">-0.870211342331981</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">0.265369870793696</td><td style="border: 1px solid white;">0.441561305121947</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"TotalCharges"</b></td><td style="border: 1px solid white;">0.961642499724251</td><td style="border: 1px solid white;">-0.231798760869362</td><td style="border: 1px solid white;">7032.0</td><td style="border: 1px solid white;">2283.30044084187</td><td style="border: 1px solid white;">2266.77136188314</td><td style="border: 1px solid white;">18.8</td><td style="border: 1px solid white;">84.445</td><td style="border: 1px solid white;">402.683333333333</td><td style="border: 1px solid white;">1397.475</td><td style="border: 1px solid white;">3798.2375</td><td style="border: 1px solid white;">5985.4476923077</td><td style="border: 1px solid white;">8684.8</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"tenure"</b></td><td style="border: 1px solid white;">0.239539749561985</td><td style="border: 1px solid white;">-1.38737163597169</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">32.3711486582422</td><td style="border: 1px solid white;">24.5594810230945</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">2.0</td><td style="border: 1px solid white;">9.0</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">55.0</td><td style="border: 1px solid white;">69.0</td><td style="border: 1px solid white;">72.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Dependents"</b></td><td style="border: 1px solid white;">0.87519857729972</td><td style="border: 1px solid white;">-1.2343780571695</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">0.299588243646173</td><td style="border: 1px solid white;">0.458110167510015</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"MonthlyCharges"</b></td><td style="border: 1px solid white;">-0.220524433943982</td><td style="border: 1px solid white;">-1.25725969454951</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">64.7616924605992</td><td style="border: 1px solid white;">30.0900470976785</td><td style="border: 1px solid white;">18.25</td><td style="border: 1px solid white;">20.05</td><td style="border: 1px solid white;">35.5</td><td style="border: 1px solid white;">70.3214285714286</td><td style="border: 1px solid white;">89.85</td><td style="border: 1px solid white;">102.6</td><td style="border: 1px solid white;">118.75</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PhoneService"</b></td><td style="border: 1px solid white;">-2.72715293844056</td><td style="border: 1px solid white;">5.43890755508706</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">0.903166264375976</td><td style="border: 1px solid white;">0.295752231783635</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PaperlessBilling"</b></td><td style="border: 1px solid white;">-0.375395747503722</td><td style="border: 1px solid white;">-1.85960618560884</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">0.592219224762175</td><td style="border: 1px solid white;">0.491456924049407</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"SeniorCitizen"</b></td><td style="border: 1px solid white;">1.83363274409285</td><td style="border: 1px solid white;">1.36259589579391</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">0.162146812437882</td><td style="border: 1px solid white;">0.368611605610013</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Partner"</b></td><td style="border: 1px solid white;">0.0679223834263394</td><td style="border: 1px solid white;">-1.9959534211947</td><td style="border: 1px solid white;">7043.0</td><td style="border: 1px solid white;">0.483032798523357</td><td style="border: 1px solid white;">0.499747510719987</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">0.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">1.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[37]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You can use describe which will compute different information according to the input method.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>mean</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>std</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>min</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>25%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>50%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>75%</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>max</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Churn</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0.265369870793696</td><td style="border: 1px solid white;">0.441561305121947</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>TotalCharges</b></td><td style="border: 1px solid white;">7032</td><td style="border: 1px solid white;">2283.30044084187</td><td style="border: 1px solid white;">2266.77136188314</td><td style="border: 1px solid white;">18.8</td><td style="border: 1px solid white;">402.683333333333</td><td style="border: 1px solid white;">1397.475</td><td style="border: 1px solid white;">3798.2375</td><td style="border: 1px solid white;">8684.8</td><td style="border: 1px solid white;">6530.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>tenure</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">32.3711486582422</td><td style="border: 1px solid white;">24.5594810230945</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">9</td><td style="border: 1px solid white;">29</td><td style="border: 1px solid white;">55</td><td style="border: 1px solid white;">72</td><td style="border: 1px solid white;">73.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Dependents</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0.299588243646173</td><td style="border: 1px solid white;">0.458110167510015</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>MonthlyCharges</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">64.7616924605992</td><td style="border: 1px solid white;">30.0900470976785</td><td style="border: 1px solid white;">18.25</td><td style="border: 1px solid white;">35.5</td><td style="border: 1px solid white;">70.3214285714286</td><td style="border: 1px solid white;">89.85</td><td style="border: 1px solid white;">118.75</td><td style="border: 1px solid white;">1585.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>PhoneService</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0.903166264375976</td><td style="border: 1px solid white;">0.295752231783635</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>PaperlessBilling</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0.592219224762175</td><td style="border: 1px solid white;">0.491456924049407</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>SeniorCitizen</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0.162146812437882</td><td style="border: 1px solid white;">0.368611605610013</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>Partner</b></td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0.483032798523357</td><td style="border: 1px solid white;">0.499747510719987</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2.0</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[38]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">describe</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;categorical&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>dtype</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>unique</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>count</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>top</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>top_percent</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PaymentMethod"</b></td><td style="border: 1px solid white;">varchar(50)</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">Electronic check</td><td style="border: 1px solid white;">33.579</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"OnlineBackup"</b></td><td style="border: 1px solid white;">varchar(38)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">43.845</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"gender"</b></td><td style="border: 1px solid white;">varchar(20)</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">50.476</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Churn"</b></td><td style="border: 1px solid white;">boolean</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">73.463</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"StreamingTV"</b></td><td style="border: 1px solid white;">varchar(38)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">39.898</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"TotalCharges"</b></td><td style="border: 1px solid white;">numeric(9,3)</td><td style="border: 1px solid white;">6530</td><td style="border: 1px solid white;">7032</td><td style="border: 1px solid white;">20.2</td><td style="border: 1px solid white;">0.156</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Contract"</b></td><td style="border: 1px solid white;">varchar(28)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">55.019</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"tenure"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">73</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">8.704</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"DeviceProtection"</b></td><td style="border: 1px solid white;">varchar(38)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">43.944</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"StreamingMovies"</b></td><td style="border: 1px solid white;">varchar(38)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">39.543</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"InternetService"</b></td><td style="border: 1px solid white;">varchar(22)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">43.959</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Dependents"</b></td><td style="border: 1px solid white;">boolean</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">70.041</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"MonthlyCharges"</b></td><td style="border: 1px solid white;">numeric(8,3)</td><td style="border: 1px solid white;">1585</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">20.05</td><td style="border: 1px solid white;">0.866</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PhoneService"</b></td><td style="border: 1px solid white;">boolean</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">90.317</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"TechSupport"</b></td><td style="border: 1px solid white;">varchar(38)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">49.311</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PaperlessBilling"</b></td><td style="border: 1px solid white;">boolean</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">59.222</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"SeniorCitizen"</b></td><td style="border: 1px solid white;">int</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">83.785</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Partner"</b></td><td style="border: 1px solid white;">boolean</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">51.697</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"MultipleLines"</b></td><td style="border: 1px solid white;">varchar(100)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">48.133</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"OnlineSecurity"</b></td><td style="border: 1px solid white;">varchar(38)</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">49.666</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"customerID"</b></td><td style="border: 1px solid white;">varchar(20)</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">7043</td><td style="border: 1px solid white;">0002-ORFBO</td><td style="border: 1px solid white;">0.014</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[43]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>All the aggregations can also be called using many built-in methods. You can for example compute the 'avg' of all the numerical columns in one line.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">avg</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Churn"</b></td><td style="border: 1px solid white;">0.265369870793696</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"TotalCharges"</b></td><td style="border: 1px solid white;">2283.30044084187</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"tenure"</b></td><td style="border: 1px solid white;">32.3711486582422</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Dependents"</b></td><td style="border: 1px solid white;">0.299588243646173</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"MonthlyCharges"</b></td><td style="border: 1px solid white;">64.7616924605992</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PhoneService"</b></td><td style="border: 1px solid white;">0.903166264375976</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"PaperlessBilling"</b></td><td style="border: 1px solid white;">0.592219224762175</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"SeniorCitizen"</b></td><td style="border: 1px solid white;">0.162146812437882</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>"Partner"</b></td><td style="border: 1px solid white;">0.483032798523357</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[39]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Or just the 'median' of a specific column.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="p">[</span><span class="s2">&quot;tenure&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[42]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>29.0</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>It is also possible to use the 'groupby' method to compute customized aggregations.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vdf</span><span class="o">.</span><span class="n">groupby</span><span class="p">([</span><span class="s2">&quot;gender&quot;</span><span class="p">,</span>
             <span class="s2">&quot;Contract&quot;</span><span class="p">],</span>
            <span class="p">[</span><span class="s2">&quot;AVG(Churn::int) AS churn&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">6</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>gender</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Contract</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>churn</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">One year</td><td style="border: 1px solid white;">0.104456824512535</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">Two year</td><td style="border: 1px solid white;">0.0305882352941176</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">0.416923076923077</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">One year</td><td style="border: 1px solid white;">0.120529801324503</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">Two year</td><td style="border: 1px solid white;">0.0260355029585799</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>5</b></td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">0.437402597402597</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[46]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: groupby, Number of rows: 6, Number of columns: 3</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Aggregations can be directly used to understand data.</p>
<p>Another way of using the power of aggregations are graphics. Our next Chapter will show you how drawing graphics in Vertica ML Python.</p>

</div>
</div>
</div>