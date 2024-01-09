# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Created by: Umar Farooq Ghumman

# Loading verticapy by default
# # All RST files will have the following at the top:
rst_prolog = """
.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib","matplotlib") 

.. raw:: html

    <div class="loader-container" id="loaderContainer">
    <img src="_static/loader.gif" alt="Loading..." class="gif-loader">
    </div>
    <div id="content" style="display:none;">
    <!-- Your page content goes here -->
    </div>

    <script>
    window.addEventListener("load", function() {
        var loader = document.getElementById("loaderContainer");
        var content = document.getElementById("content");

        // Hide the loader and show the content after the page has loaded
        loader.style.display = "none";
        content.style.display = "block";
    });
    </script>

    <script>
    var previousOption = document.getElementById("filter-select").value;

    function switchVersion(selectedVersion) {
        var currentURL = window.location.href;
        
        var newURL = currentURL.replace(previousOption, selectedVersion);
        
        window.location.href = newURL;
        
        // Update the previousOption with the new selected version
        previousOption = selectedVersion;
    }
    </script>
    <div id="main">
    <button class="openbtn" onclick="openNav()">Feedback</button>  
    </div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

    <div id="mySidebar" class="feedback_sidebar">
        <a href="javascript:void(0)" class="closebtn" onclick="closeNav()">×</a>
        <div id="feedback-form">
            <form method='post' enctype='multipart/form-data' id='gform_50' action='https://vertica.com/python/thank-you.php'>
                <div class='gform_body'>
                    <ul id='gform_fields_50' class='gform_fields top_label form_sublabel_below description_below'>
                        <li class='feedback_title'>
                            Feedback
                        </li>
                        <li id='field_50_1'
                            class='gfield gfield_contains_required field_sublabel_below field_description_below gfield_visibility_visible'>
                            <label class='gfield_label'>Was this page helpful?<span class='gfield_required'>*</span></label>
                            <div class='ginput_container ginput_container_radio'>
                                <ul class='gfield_radio' id='input_50_1'>
                                    <li class='gchoice_50_1_0'><input name='input_1' type='radio' value='yes' id='choice_50_1_0' tabindex='100'/><label for='choice_50_1_0'
                                                                                                                id='label_50_1_0'>Yes</label>
                                    </li>
                                    <li class='gchoice_50_1_1'><input name='input_1' type='radio' value='no' id='choice_50_1_1'
                                                                    tabindex='101'/><label for='choice_50_1_1' id='label_50_1_1'>No</label>
                                    </li>
                                </ul>
                            </div>
                        </li>
                        <li id='field_50_2' class='gfield field_sublabel_below field_description_below gfield_visibility_visible'>
                            <label class='gfield_label'>Do you have any additional feedback?</label>
                            <div class='ginput_container ginput_container_radio'>
                                <ul class='gfield_radio' id='input_50_2'>
                                    <li class='gchoice_50_2_0'><input name='input_2' type='radio' value='yes' id='choice_50_2_0'
                                                                    tabindex='102'/><label for='choice_50_2_0' id='label_50_2_0'>Yes</label>
                                    </li>
                                    <li class='gchoice_50_2_1'><input name='input_2' type='radio' value='no' id='choice_50_2_1' tabindex='103'/><label for='choice_50_2_1'
                                                                                                                id='label_50_2_1'>No</label>
                                    </li>
                                </ul>
                            </div>
                        </li>
                        <li id='field_50_3'
                            class='gfield field_sublabel_below field_description_below hidden_label gfield_visibility_visible' style="display:none;">
                            <label class='gfield_label' for='input_50_3'>Additional Feedback</label>
                            <div class='ginput_container ginput_container_textarea'><textarea name='input_3' id='input_50_3'
                                                                                            class='textarea medium' tabindex='104'
                                                                                            maxlength='600'
                                                                                            placeholder='Feedback, comments, etc.'
                                                                                            aria-invalid="false" rows='10'
                                                                                            cols='50'></textarea></div>
                        </li>
                    </ul>
                </div>
                <div class='gform_footer top_label'><input type='submit' id='gform_submit_button_50' disabled class='gform_button button'
                                                        value='Submit' tabindex='105'
                                                        onclick='if(window["gf_submitting_50"]){return false;}  window["gf_submitting_50"]=true;  '
                                                        onkeypress='if( event.keyCode == 13 ){ if(window["gf_submitting_50"]){return false;} window["gf_submitting_50"]=true;  jQuery("#gform_50").trigger("submit",[true]); }'/>
                    <input type='hidden' class='gform_hidden' name='is_submit_50' value='1'/>
                    <input type='hidden' class='gform_hidden' name='gform_submit' value='50'/>

                    <input type='hidden' class='gform_hidden' name='gform_unique_id' value=''/>
                    <input type='hidden' class='gform_hidden' name='state_50'
                        value='WyJbXSIsIjEzNjNmNTlmM2QzNWRlZjNlNTk2NGE5ODc2M2U2YWFjIl0='/>
                    <input type='hidden' class='gform_hidden' name='gform_target_page_number_50' id='gform_target_page_number_50'
                        value='0'/>
                    <input type='hidden' class='gform_hidden' name='gform_source_page_number_50' id='gform_source_page_number_50'
                        value='1'/>
                    <input type='hidden' name='gform_field_values' value=''/>

                </div>
            </form>
        </div>
    </div>

    <script>
    function openNav() {
    document.getElementById("mySidebar").style.width = "250px";
    document.getElementById("main").style.marginLeft = "250px";
    }

    function closeNav() {
    document.getElementById("mySidebar").style.width = "0";
    document.getElementById("main").style.marginLeft= "0";
    }
    </script>

    <script type='text/javascript'>
        (function ($) {
            // Enable/disable submit button
            $('#choice_50_1_0, #choice_50_1_1').on('change', function () {
                $('#feedback-form .gform_button').prop('disabled', !$(this).val());
            });

            // Show/hide additional comments
            $('#choice_50_2_0, #choice_50_2_1').on('change', function () {
                if ($(this).val() === 'yes') {
                    $('#field_50_3').slideDown('fast');
                } else {
                    $('#field_50_3').slideUp('fast');
                }
            });

            $('#gform_50').on('submit', function(e) {
                e.preventDefault();
                var $form = $(this);

                $form.css({
                    'opacity': .5,
                    'cursor': 'progress',
                });

                $.ajax({
                    type: "post",
                    dataType: "json",
                    url: 'https://www.vertica.com/wp-admin/admin-ajax.php',
                    data: {
                        action: 'python_feedback',
                        formData: $(this).serializeArray(),
                    },
                    success: function (response) {
                        if (response.is_valid) {
                            $form.html(response.confirmation_message);
                        } else {
                            console.log('something went wrong');
                        }

                        console.log( response );
                        $form.css({
                            'opacity': 1,
                            'cursor': 'default',
                        });
                    }
                });
            });
        })(jQuery);
    </script>
    
    
"""


import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "VerticaPy"
copyright = "2023-2024 OpenText. All rights reserved."
author = "Vertica"
release = "1.0.x"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx_inline_tabs",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.graphviz",
]

templates_path = ["_templates"]

exclude_patterns = []

import verticapy  # isort:skip

# version = '%s r%s' % (pandas.__version__, svn_version())
version = str(verticapy.__version__)

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "furo"  #'pydata_sphinx_theme'

html_static_path = ["_static"]


html_logo = "_static/Vertica-OT.svg"


# Theme Options for Furo theme

html_theme_options = {
    "footer_icons": [
        {
            "name": "Privacy Policy",
            "url": "https://www.opentext.com/about/privacy",
            "html": "Privacy Policy",
            "class": "bottom_buttons",
        },
        {
            "name": "Cookies Policy",
            "url": "https://www.opentext.com/about/cookie-policy",
            "html": "Cookies Policy",
            "class": "bottom_buttons",
        },
    ],
    "announcement": f"""<div class='centered-content'>
                            
                        </div>



                    <div class="main-header-container">

                        <div class="very-top-container">

                            <a href="./home.html">
                            
                                <svg id="Logo_Layer_2" data-name="Logo_Layer 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 282.57 29.89" height="30px" width="180px" fill="var(--color-announcement-text)">
                                <g id="Logo_Layer_1-2" data-name="Logo_Layer 1">
                                    <g>
                                    <g>
                                        <path d="M169.66,5.16h2.74l5.71,15.65,5.81-15.65h2.6l-7.21,18.78h-2.45l-7.21-18.78Z"></path>
                                        <path d="M198.52,20.13c-.23,1.31-.81,2.35-1.75,3.1-.94,.75-2.28,1.13-4.04,1.13-2.07,0-3.66-.63-4.76-1.88s-1.66-3-1.66-5.25c0-1.17,.16-2.22,.47-3.13,.32-.91,.76-1.67,1.33-2.29,.57-.61,1.25-1.08,2.04-1.41,.79-.32,1.65-.49,2.58-.49,1.05,0,1.96,.17,2.71,.5,.75,.33,1.37,.79,1.85,1.38s.84,1.28,1.08,2.09c.24,.81,.35,1.69,.35,2.66v1.16h-10.1c0,.68,.09,1.32,.28,1.91,.18,.59,.45,1.1,.79,1.53s.77,.77,1.28,1.01c.51,.25,1.1,.37,1.76,.37,1.14,0,1.97-.21,2.5-.64,.53-.43,.89-1.01,1.08-1.75h2.21Zm-2.08-4.13c0-.65-.07-1.24-.21-1.76-.14-.53-.36-.98-.66-1.35s-.68-.67-1.16-.88c-.47-.21-1.04-.32-1.71-.32-1.21,0-2.15,.37-2.83,1.1-.68,.74-1.07,1.81-1.2,3.21h7.76Z"></path>
                                        <path d="M201.38,10.34h2.26v2.1c.51-.75,1.11-1.33,1.79-1.74,.68-.4,1.41-.6,2.18-.6,.47,0,.88,.04,1.21,.11l-.29,2.13c-.18-.04-.35-.07-.53-.09-.18-.03-.37-.04-.58-.04-.42,0-.85,.08-1.29,.22-.44,.15-.85,.4-1.24,.75-.39,.33-.69,.72-.91,1.16-.22,.44-.33,.95-.33,1.53v8.07h-2.29V10.34Z"></path>
                                        <path d="M212.21,12.16h-2.1v-1.81h2.1v-3.71h2.29v3.71h3.6v1.81h-3.6v7.97c0,.67,.15,1.2,.45,1.6s.75,.6,1.37,.6c.38,0,.72-.03,1.01-.08,.29-.05,.56-.12,.8-.21l.39,1.76c-.35,.12-.73,.22-1.14,.3-.41,.08-.9,.12-1.46,.12-.77,0-1.39-.11-1.87-.34-.47-.23-.85-.55-1.13-.97-.28-.42-.47-.92-.57-1.5-.1-.58-.14-1.21-.14-1.89v-7.36Z"></path>
                                        <path d="M220.31,5.16h2.45v2.66h-2.45v-2.66Zm.08,5.18h2.29v13.6h-2.29V10.34Z"></path>
                                        <path d="M237.54,19.34c-.23,1.56-.83,2.79-1.79,3.68-.96,.89-2.32,1.34-4.08,1.34-1.04,0-1.95-.17-2.75-.5-.8-.33-1.46-.81-2-1.43-.54-.62-.94-1.38-1.21-2.26-.27-.89-.41-1.88-.41-2.99s.15-2.08,.45-2.97,.73-1.66,1.29-2.29c.56-.63,1.25-1.12,2.05-1.47,.81-.35,1.72-.53,2.73-.53,1.63,0,2.89,.4,3.79,1.2,.89,.8,1.46,1.9,1.71,3.3l-2.29,.42c-.09-.44-.21-.84-.35-1.21-.15-.37-.36-.69-.62-.96-.26-.27-.58-.48-.96-.63-.38-.15-.83-.22-1.35-.22-.72,0-1.33,.14-1.84,.41-.51,.27-.93,.65-1.25,1.13-.32,.48-.56,1.05-.71,1.7-.15,.65-.22,1.34-.22,2.08s.07,1.43,.21,2.08c.14,.65,.37,1.21,.68,1.7,.32,.48,.73,.86,1.24,1.14,.51,.28,1.13,.42,1.87,.42,1.1,0,1.92-.28,2.46-.84,.53-.56,.88-1.32,1.04-2.29h2.31Z"></path>
                                        <path d="M248.73,22.07c-.49,.6-1.11,1.11-1.87,1.54-.75,.43-1.69,.64-2.81,.64-.68,0-1.31-.08-1.88-.25-.57-.17-1.06-.42-1.47-.76-.41-.34-.73-.77-.96-1.28-.23-.51-.34-1.1-.34-1.79,0-.93,.2-1.68,.59-2.26,.39-.58,.92-1.03,1.58-1.37,.66-.33,1.4-.57,2.22-.71,.82-.14,1.67-.23,2.55-.26l2.34-.08v-.53c0-1.16-.25-1.99-.74-2.5-.49-.51-1.29-.76-2.39-.76s-1.89,.18-2.39,.54c-.51,.36-.84,.85-1,1.49l-2.08-.24c.25-1.25,.83-2.15,1.76-2.72s2.19-.85,3.79-.85c.98,0,1.81,.12,2.49,.35,.67,.24,1.22,.57,1.64,1.01,.42,.44,.72,.98,.91,1.63,.18,.65,.28,1.38,.28,2.18v8.84h-2.21v-1.87Zm-.05-4.94l-2.18,.08c-.93,.04-1.7,.12-2.31,.26-.61,.14-1.1,.33-1.46,.57s-.61,.53-.76,.88c-.15,.35-.22,.75-.22,1.21,0,.68,.21,1.23,.62,1.64,.41,.41,1.09,.62,2.04,.62,1.35,0,2.4-.33,3.16-1,.75-.67,1.13-1.58,1.13-2.73v-1.53Z"></path>
                                        <path d="M254.4,5.16h6.29c1.38,0,2.53,.15,3.43,.45,.9,.3,1.61,.7,2.13,1.21,.52,.51,.88,1.1,1.08,1.77,.2,.68,.3,1.39,.3,2.14s-.11,1.54-.32,2.26c-.21,.72-.58,1.35-1.1,1.89-.53,.54-1.24,.97-2.13,1.29-.89,.32-2.02,.47-3.39,.47h-3.89v7.29h-2.39V5.16Zm6.26,9.44c.91,0,1.67-.09,2.26-.26,.6-.17,1.06-.43,1.39-.76,.33-.33,.57-.73,.7-1.2,.13-.46,.2-.99,.2-1.56s-.07-1.12-.21-1.58c-.14-.46-.39-.83-.74-1.13-.35-.3-.82-.52-1.41-.67-.59-.15-1.33-.22-2.22-.22h-3.84v7.39h3.87Z"></path>
                                        <path d="M270.71,26.83c.49,.17,.98,.26,1.47,.26,.56,0,.99-.14,1.3-.43,.31-.29,.61-.83,.91-1.62l.55-1.45-5.76-13.26h2.66l4.29,10.68,3.92-10.68h2.52l-5.94,14.89c-.25,.61-.49,1.15-.74,1.62-.25,.46-.52,.86-.82,1.2-.3,.33-.65,.58-1.07,.75-.41,.17-.92,.25-1.51,.25-.42,0-.8-.03-1.13-.08-.33-.05-.7-.15-1.1-.29l.45-1.84Z"></path>
                                    </g>
                                    <g>
                                        <path d="M20.11,14.32c0,4.73-2.72,9.86-10.04,9.86C4.8,24.17,0,21.23,0,14.32,0,8.58,3.62,4.17,10.79,4.57c7.63,.43,9.32,6.34,9.32,9.75Zm-13.62-3.6c-.68,1.05-.93,2.31-.93,3.57,0,2.9,1.43,5.53,4.52,5.53s4.44-2.42,4.44-5.29c0-2.03-.5-3.6-1.54-4.58-1.15-1.05-2.44-1.08-3.23-1.01-1.58,.1-2.51,.63-3.26,1.78Z"></path>
                                        <path d="M69.03,6.68c.54-.57,.9-1,1.86-1.47,.86-.36,2.08-.64,3.4-.64,1.11,0,2.37,.18,3.33,.72,1.97,1.04,2.47,2.72,2.47,5.63v12.79h-5.48V13.17c0-1.68-.04-2.29-.25-2.8-.43-1-1.4-1.43-2.47-1.43-2.9,0-2.9,2.33-2.9,4.66v10.11h-5.52V5.03h5.56v1.65Z"></path>
                                        <path d="M60.85,18.87c-.51,1.07-2.13,5.3-9.32,5.3-5.56,0-9.46-3.4-9.46-9.53,0-4.52,2.22-10.08,9.61-10.08,1.11,0,4.34-.14,6.74,2.44,2.44,2.62,2.58,6.27,2.65,8.39h-13.37c-.04,2.33,1.29,4.66,4.26,4.66s4.05-1.97,4.73-3.3l4.16,2.11Zm-5.48-6.92c-.11-.79-.25-1.83-1.04-2.65-.68-.68-1.72-1.04-2.69-1.04-1.33,0-2.26,.65-2.8,1.22-.75,.82-.93,1.68-1.11,2.47h7.63Z"></path>
                                        <path d="M137.59,8.91h3.66v-3.88h-3.62V1.1h-5.48v3.93h-.87l-2.51,3.88h3.38v8.88c0,1.79,.04,3.12,.79,4.23,1.18,1.72,3.26,1.83,5.16,1.83,1,0,1.72-.11,2.9-.29v-4.3l-1.97,.07c-1.47,0-1.47-.97-1.43-2.11V8.91Z"></path>
                                        <path d="M82.85,1.1h5.48v3.93h5.7l-2.51,3.88h-3.22v8.31c-.04,1.15-.04,2.11,1.43,2.11l1.97-.07v4.3c-1.18,.18-1.9,.29-2.9,.29-1.9,0-3.98-.11-5.16-1.83-.75-1.11-.79-2.44-.79-4.23V1.1Z"></path>
                                        <path d="M146.33,8.91h-.68v-3.3h-1.25v-.57h3.15v.57h-1.23v3.3Zm5.31,0h-.67l.02-2.39,.04-.79-.19,.65-.79,2.53h-.61l-.78-2.53-.2-.64,.05,.78,.02,2.39h-.65v-3.88h.92l.97,3.06,.95-3.06h.92v3.88Z"></path>
                                        <path d="M111.31,18.89l-1.05,1.62c-.98,1.44-2.83,3.66-8.13,3.66s-9.29-3.4-9.29-9.53c0-4.52,2.22-10.08,9.61-10.08,1.11,0,4.34-.14,6.74,2.44,2.44,2.62,2.58,6.27,2.65,8.39h-13.37c-.04,2.33,1.12,4.66,4.09,4.66s3.88-1.97,4.56-3.3l4.19,2.13Zm-5.17-6.93c-.11-.79-.25-1.83-1.04-2.65-.68-.68-1.72-1.04-2.69-1.04-1.33,0-2.26,.65-2.8,1.22-.75,.82-.93,1.68-1.11,2.47h7.63Z"></path>
                                        <polygon points="130.09 23.71 123.24 13.88 128.96 5.03 122.68 5.03 119.98 9.21 117.07 5.03 110.79 5.03 116.96 13.88 110.61 23.71 116.88 23.71 120.22 18.55 123.82 23.71 130.09 23.71"></polygon>
                                        <path d="M27.69,6.68c.53-.8,2.22-2.11,4.91-2.11,4.59,0,7.81,3.48,7.81,9.71,0,3.83-1.4,9.89-8.05,9.89-2.39,0-4.24-1.33-4.71-2.11v7.83h-5.52V5.03h5.56v1.65Zm3.73,2.19c-1,0-2.04,.4-2.83,1.4-.79,.97-1.15,2.47-1.15,4.1,0,2.2,.65,3.54,1.33,4.3,.64,.72,1.58,1.13,2.47,1.13,2.69,0,3.87-2.83,3.87-5.57,0-2.3-.68-4.74-2.94-5.27-.25-.07-.5-.1-.75-.1Z"></path>
                                    </g>
                                    <rect x="159.84" width="1" height="29.13"></rect>
                                    </g>
                                </g>
                                </svg>
                            </a>

                            <div class="top-dropdown">
                                <button class="dropdown-btn">&#9776;</button>
                                <div class="dropdown-content">
                                    <a class="top-button" href="./getting_started.html" id="sitenav-solutions">Getting Started</a>
                                    <a class="top-button" href="./user_guide.html" id="sitenav-solutions">User Guide</a>
                                    <a class="top-button" href="./api.html" id="sitenav-solutions">API Reference</a>
                                    <a class="top-button" href="./examples.html" id="sitenav-solutions">Examples</a>
                                </div>
                            </div>
                            <div class="search-top-dropdown">
                                <button class="search-dropdown-btn">
                                    <svg width="30" height="30" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#000000" stroke-width="10">
                                    <!-- Handle -->
                                    <line x1="60" y1="60" x2="85" y2="85"></line>
                                    <!-- Glass -->
                                    <circle cx="35" cy="35" r="30"></circle>
                                    </svg>
                                </button>
                                <div class="search-dropdown-content">
                                    <form class="sidebar-search-container_top-2" method="get" action="search.html" role="search">
                                    <input class="sidebar-search" placeholder="Search" name="q" aria-label="Search">
                                    <input type="hidden" name="check_keywords" value="yes">
                                    <input type="hidden" name="area" value="default">
                                    </form>
                                </div>
                            </div>
                            <div class="right-top-container">
                                <div class='top_search'>
                                    <form class='sidebar-search-container sidebar-search-container_top' method='get' action='search.html' role='search'>
                                        <input class='sidebar-search' placeholder='Search' name='q' aria-label='Search'>
                                        <input type='hidden' name='check_keywords' value='yes'>
                                        <input type='hidden' name='area' value='default'>
                                    </form>
                                </div>
                                <div class="form-group">
                                    <select class="form-control" id="filter-select" name="filter-select" onchange="switchVersion(this.value)">
                                        <option value="1.0.x" selected="">1.0.x</option>
                                    </select>
                                </div>
                                <div class="color-theme-container">
                                    <button class="theme-toggle">
                                        <div class="visually-hidden">Toggle Light / Dark / Auto color theme</div>
                                        <svg class="theme-icon-when-auto"><use href="#svg-sun-half"></use></svg>
                                        <svg class="theme-icon-when-dark"><use href="#svg-moon"></use></svg>
                                        <svg class="theme-icon-when-light"><use href="#svg-sun"></use></svg>
                                    </button>
                                </div>
                            </div>
                            
                        </div>
                        <div class="top-bottom-container">

                            <div class="top-button-container">
                                <a class="top-button" href="./index.html" id="sitenav-solutions">Home</a>
                                <a class="top-button" href="./user_guide.html" id="sitenav-solutions">User Guide</a>
                                <a class="top-button" href="./api.html" id="sitenav-solutions">API Reference</a>
                                <a class="top-button" href="./examples.html" id="sitenav-solutions">Examples</a>
                            </div>

                            <div class="right-content">
                                <button class="github-button" onclick="window.open('https://github.com/vertica/VerticaPy', '_blank')">
                                    <span class="button-text">View on GitHub</span>
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" width="20px" height="20px" fill="white">

                                        <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"></path></svg>
                                </button>
                            </div>
                    </div>
                </div>""",
    "light_css_variables": {
        "color-announcement-background": "white",
        "color-announcement-text": "#101c2f",
    },
}

html_favicon = "_static/ot_favicon.svg"
# Customization
html_css_files = [
    "css/custom_styling.css",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]
