docker build -t rajib/kim-tc .
nvidia-docker run -it -v /mnt/b7e83844-6bc7-43cd-8eee-87edc0eadf1e:/mnt/b7e83844-6bc7-43cd-8eee-87edc0eadf1e -v /mnt/58a96951-a863-483e-b6bd-01c265d94667/:/mnt/58a96951-a863-483e-b6bd-01c265d94667/ -v /mnt/1b47d000-2aee-4aa6-92a0-ff08c97e14fc:/mnt/1b47d000-2aee-4aa6-92a0-ff08c97e14fc rajib/kim-tc bash
