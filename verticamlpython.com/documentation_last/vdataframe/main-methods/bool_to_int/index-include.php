<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.bool_to_int">vDataFrame.bool_to_int<a class="anchor-link" href="#vDataFrame.bool_to_int">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">bool_to_int</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Converts all the booleans vcolumns to integers.</p>

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
<div class="prompt input_prompt">In&nbsp;[107]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vDataFrame</span>
<span class="n">churn</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.churn&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">churn</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PaymentMethod</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>OnlineBackup</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>gender</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Churn</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>StreamingTV</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>TotalCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Contract</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>tenure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>DeviceProtection</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>StreamingMovies</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Dependents</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MonthlyCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PhoneService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>TechSupport</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PaperlessBilling</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SeniorCitizen</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Partner</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MultipleLines</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>OnlineSecurity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>customerID</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mailed check</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">593.300</td><td style="border: 1px solid white;">One year</td><td style="border: 1px solid white;">9</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">65.600</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0002-ORFBO</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Mailed check</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">542.400</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">9</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">59.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0003-MKNFE</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Electronic check</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">280.850</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">73.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0004-TLHLJ</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Electronic check</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">1237.850</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">13</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">98.000</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0011-IGKFF</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Mailed check</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">267.400</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">83.900</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">True</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0013-EXCHZ</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: churn, Number of rows: 7043, Number of columns: 21
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[108]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">churn</span><span class="o">.</span><span class="n">bool_to_int</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PaymentMethod</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>OnlineBackup</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>gender</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Churn</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>StreamingTV</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>TotalCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Contract</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>tenure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>DeviceProtection</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>StreamingMovies</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>InternetService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Dependents</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MonthlyCharges</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PhoneService</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>TechSupport</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>PaperlessBilling</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>SeniorCitizen</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Partner</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>MultipleLines</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>OnlineSecurity</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>customerID</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mailed check</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">593.300</td><td style="border: 1px solid white;">One year</td><td style="border: 1px solid white;">9</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">65.600</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0002-ORFBO</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Mailed check</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">542.400</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">9</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">DSL</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">59.900</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0003-MKNFE</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Electronic check</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">280.850</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">73.900</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0004-TLHLJ</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Electronic check</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Male</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">1237.850</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">13</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">98.000</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0011-IGKFF</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Mailed check</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Female</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">267.400</td><td style="border: 1px solid white;">Month-to-month</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">Fiber optic</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">83.900</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">Yes</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">No</td><td style="border: 1px solid white;">0013-EXCHZ</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[108]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: churn, Number of rows: 7043, Number of columns: 21</pre>
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
    <tr><td><a href="../astype">vDataFrame.astype</a></td> <td>Converts the vcolumns to the input types.</td></tr>
</table>
</div>
</div>
</div>