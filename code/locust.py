from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def hit_root(self):
        self.client.get("/")

    @task(7)
    def download_data(self):
        self.client.get("/download")