"""One build fingerprint prevents mixed Phase 4A/4B assets after deploy."""
import re
import unittest
from web.server import create_app


class BuildTests(unittest.TestCase):
    def test_served_page_assets_and_backend_have_one_build(self):
        app=create_app()
        with app.test_client() as client:
            page=client.get('/')
            build=client.get('/health').json['build']
            self.assertTrue(build.startswith('4b-debug-'))
            self.assertIn(build.encode(),page.data)
            self.assertEqual('no-store',page.headers['Cache-Control'])
            self.assertIn(b'<details id="medicine-panel"',page.data)
            self.assertIn(b'medicine-identification.js',page.data)
            self.assertNotIn(b'Discard image',page.data)
            for url in re.findall(r'(?:src|href)="(/static/[^"]+)"',page.text):
                self.assertTrue(url.endswith('?v='+build),url)
                response=client.get(url)
                self.assertEqual(200,response.status_code)
                self.assertEqual('no-store',response.headers['Cache-Control'])
                self.assertEqual(build,response.headers['X-Visual-Assistant-Build'])
                response.close()
            self.assertEqual(400,client.post('/api/medicine/identify-crop').status_code)
