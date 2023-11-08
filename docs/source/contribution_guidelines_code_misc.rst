.. _contribution_guidelines.code.misc:


===============
Miscellenious
===============

Code formatting as per PEP 8
----------------------------------

Once you are satisfied with your code, please run `black <https://black.readthedocs.io/en/stable/>`_ for your code. Black will automatically format all your code to make it professional and consistent with PEP 8.

Next please run `pylint <https://pypi.org/project/pylint/>`_ and ensure that your score is above the minimum threshold of 5. Pylint will automatically provide you with the improvement opportunities that you can adjust to increaes your score.

As per the updated CI/CD, no code will be accepted that requires formatting using black or has a lower pylint score than the threshold stated above. 

License Headers
--------------------

Every file in this project must use the following Apache 2.0 header (with the appropriate year or years in the "[yyyy]" box; if a copyright statement from another party is already present in the code, you may add the statement on top of the existing copyright statement):

.. code-block:: python

    """
    Copyright  (c)  2018-2023 Open Text  or  one  of its
    affiliates.  Licensed  under  the   Apache  License,
    Version 2.0 (the  "License"); You  may  not use this
    file except in compliance with the License.

    You may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0

    Unless  required  by applicable  law or  agreed to in
    writing, software  distributed  under the  License is
    distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
    OR CONDITIONS OF ANY KIND, either express or implied.
    See the  License for the specific  language governing
    permissions and limitations under the License.
    """
