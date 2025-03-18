from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # Simulate real users with random wait times

    @task
    def load_test(self):
        self.client.get("/")  # Simulating a user hitting the root endpoint
