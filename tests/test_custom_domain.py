import os
import re
import unittest

class TestCustomDomainConfig(unittest.TestCase):
    def test_cname_file_exists_and_valid(self):
        self.assertTrue(os.path.isfile("CNAME"), "CNAME file must exist in project root")
        with open("CNAME") as f:
            content = f.read().strip()
        self.assertEqual(content, "blog.aprhisiart.uk", "CNAME must contain blog.aprhisiart.uk")

    def test_quarto_yml_config(self):
        self.assertTrue(os.path.isfile("_quarto.yml"), "_quarto.yml must exist")
        with open("_quarto.yml") as f:
            content = f.read()
        self.assertRegex(content, r"resources:\s*\n\s*-\s*CNAME", "resources must include CNAME")
        self.assertRegex(content, r"site-url:\s*[\"']?https://blog\.aprhisiart\.uk[\"']?", "site-url must match custom domain")

    def test_rendered_output(self):
        self.assertTrue(os.path.isfile("_site/CNAME"), "_site/CNAME must be present in rendered output")
        with open("_site/CNAME") as f:
            self.assertEqual(f.read().strip(), "blog.aprhisiart.uk")

if __name__ == "__main__":
    unittest.main()
