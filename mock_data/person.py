import numpy as np
from numpy.random import choice, randint, uniform, normal, exponential
import random
import string
from datetime import datetime, timedelta
from config import n_distinct_actions, n_customers,  start_date, end_date,\
    n_legit_recipients, n_fraud_recipients


# parse dates
start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
diff_in_seconds = (end_datetime - start_datetime).total_seconds()

# mean and stds of relatively rates of customers doing general actions
fraud_rates = uniform(size=n_distinct_actions)
fraud_stds = uniform(size=n_distinct_actions)
legit_rates = uniform(size=n_distinct_actions)
legit_stds = uniform(size=n_distinct_actions)
action_names = [f"action_{i}" for i in range(n_distinct_actions)]

# account types
account_types = ['checking', 'savings', 'credit_card']  # types of accounts
account_types_rates = [0.5, 0.3, 0.2]  # rates of selecting each account type

# make list of valid recipients
legit_recipients = []
for i in range(n_legit_recipients):
    recipient = ''.join(random.choice(string.digits) for i in range(10))
    legit_recipients.append(recipient)
fraud_recipients = []
for i in range(n_fraud_recipients):
    recipient = ''.join(random.choice(string.digits) for i in range(10))
    fraud_recipients.append(recipient)


class Person():
    """
    The class will be used to generate mock session records for a specific
    customer or fraudster impersonating a customer.
    """

    def __init__(self):
        """
        Make a person.
        """

        self.age = randint(18, 70)
        # login location
        self.longitude_mean = uniform(-130, -75)
        self.longitude_std = uniform(0.1, 2)
        self.latitude_mean = uniform(30, 52)
        self.latitude_std = uniform(0.1, 2)
        self.home_longitude = normal(self.longitude_mean, self.longitude_std)
        self.home_latitude = normal(self.latitude_mean, self.latitude_std)
        # gender and marital status
        self.gender = choice(['M', 'F'])
        self.marital_status = choice(['single', 'married', 'divorced'])
        # arbitrarily making defrauded people less likely to be divorced women
        if self.fraud_label == 1 and self.marital_status == 'divorced' \
                and self.gender == 'F' and uniform() < 0.5:
            self.marital_status = choice(['single', 'married'])
        # account types
        self.account_types = choice(account_types, size=3, p=account_types_rates)
        self.account_types = np.unique(self.account_types)
        if self.fraud_label == 0:  # legit customers have a preferred account type to send from
            self.account_use_rates = uniform(size=len(self.account_types))
            self.account_use_rates /= self.account_use_rates.sum()
        else:  # fraudsters use all account types equally
            self.account_use_rates = np.ones(len(self.account_types)) / len(self.account_types)
        # preferred recipients
        self.preferred_recipient_rate = uniform()  # probability a preferred recipient is sent to
        n_recips = 1 + int(exponential(10))  # number of preferred recipients
        self.preferred_recipients = choice(legit_recipients, size=n_recips, replace=False)

    def make_customer_info_record(self):
        """Make a record with customer info."""

        record = {
            'customerId': self.customer_id,
            'age': self.age,
            'gender': self.gender,
            'maritalStatus': self.marital_status,
            'homeLongitude': self.home_longitude,
            'homeLatitude': self.home_latitude,
        }
        return record

    def make_session(self):
        """
        Make a session of events, inlcuding transactions, logins, and general actions.
        """

        session_id = ''.join(random.choice(string.ascii_letters) for i in range(10))
        # random time between start_time and end_time
        event_time = start_datetime + timedelta(seconds=diff_in_seconds * uniform())
        session = []
        n_actions = max(1, int(exponential(self.n_actions_mean)))
        for i in range(n_actions):
            # increase time by a random amount
            time_delta = timedelta(seconds=exponential(10))
            if self.fraud_label == 1:  # fraudsters are faster (more computer literate)
                time_delta *= 0.8
            elif self.age > 50:  # older people are slower
                time_delta *= 1.5
            event_time += time_delta
            # first record is always a login event
            if i == 0:
                event = self._make_login_event(session_id, event_time)
            # sometimes make a transaction
            elif uniform() < self.transaction_rate:
                event = self._make_transaction_event(session_id, event_time)
            # otherwise make a general action
            else:
                event = self._make_general_action_event(session_id, event_time)
            session.append(event)
        return session

    def _make_event(self, session_id, event_time, action):
        """
        Make a general event.

        Parameters
        ----------
        session_id : str
            ID of session.
        event_time : datetime
            Time of event.
        action : str
            Name of action.
        """

        event = {
            'fraudLabel': self.fraud_label,
            'uniqueId': ''.join(random.choice(string.ascii_letters) for i in range(16)),
            'customerId': self.customer_id,
            'sessionId': session_id,
            'timestamp': str(event_time),
            'action': action,
        }
        return event

    def _make_login_event(self, session_id, event_time):
        """
        Make a login event

        Parameters
        ----------
        session_id : str
            ID of session.
        event_time : datetime
            Time of event.
        """

        event = self._make_event(session_id, event_time, 'login')
        # add location info
        longitude, latitude = self._pick_login_location()
        event['longitude'] = longitude
        event['latitude'] = latitude
        # user_agent, etc.
        ...
        return event

    def _make_transaction_event(self, session_id, event_time):
        """
        Make a transaction event.

        Parameters
        ----------
        session_id : str
            ID of session.
        event_time : datetime
            Time of event.
        """

        event = self._make_event(session_id, event_time, 'transaction')
        # add amount
        amount = round(exponential(self.transaction_amount_mean), 2)
        event['amount'] = amount
        # add account type
        account_type = choice(self.account_types, p=self.account_use_rates)
        event['accountType'] = account_type
        # recipient
        if self.fraud_label == 1:
            recipient = choice(fraud_recipients)
        # legit customers prefer some recipients
        elif uniform() < self.preferred_recipient_rate:
            recipient = choice(self.preferred_recipients)
        else:
            recipient = choice(legit_recipients)
        event['recipient'] = recipient
        return event

    def _make_general_action_event(self, session_id, event_time):
        """
        Make a random general action record.

        Parameters
        ----------
        session_id : str
            ID of session.
        event_time : datetime
            Time of event.
        """

        # randomly pick an action based on the rate
        action_name = choice(action_names, p=self.action_rates)
        event = self._make_event(session_id, event_time, action_name)
        return event

    def _pick_login_location(self):
        """choose a random location for a login event."""

        # 10% chance of completly random location
        if uniform() < 0.1:
            longitude = uniform(-130, -75)
            latitude = uniform(30, 52)
        # 90% chance of using customer's specified mean and std
        else:
            longitude = normal(self.longitude_mean, self.longitude_std)
            latitude = normal(self.latitude_mean, self.latitude_std)
        return longitude, latitude

    # samples from a truncated normal distribution
    def truncated_normal(self, mu, sigma, a=0, b=1):
        """
        Random sample from normal distribution truncated to between a and b.
        If mu and sigma are list-like, then return an array of truncated values.

        Parameters
        ----------
        mu : float or array-like
            Mean of normal distribution.
        sigma : float or array-like
            Standard deviation of normal distribution.
        a : float or array-like
            Lower bound of truncation.
        b : float or array-like
            Upper bound of truncation.
        """

        rvs = normal(mu, sigma)
        return np.minimum(np.maximum(rvs, a), b)


class Customer(Person):

    def __init__(self, customer_id):
        """
        Make a customer.

        Parameters
        ----------
        customer_id : str
            ID of customer.
        """

        # specify fraud label
        self.fraud_label = 0
        # customer_id
        self.customer_id = customer_id
        # relative rates of each general action
        self.action_rates = self.truncated_normal(legit_rates, legit_stds)
        self.action_rates /= self.action_rates.sum()
        # mean number of actions per session
        self.n_actions_mean = exponential(20)
        # rate of transactions relative to general actions
        self.transaction_rate = self.truncated_normal(mu=0.05, sigma=0.1, a=0, b=0.5)
        # transaction amount mean (amount will be sampled from an exponential distribution)
        self.transaction_amount_mean = exponential(300)
        super().__init__()


class Fraudster(Person):

    def __init__(self):
        """
        Make a fraudster.
        """

        # specify fraud label
        self.fraud_label = 1
        # customer_id will be set in make_session for the legit customer being impersonated
        self.customer_id = None
        # relative rates of each general action
        self.action_rates = self.truncated_normal(fraud_rates, fraud_stds)
        self.action_rates /= self.action_rates.sum()
        # mean number of actions per session
        self.n_actions_mean = exponential(30)
        # rate of transactions relative to general actions
        self.transaction_rate = self.truncated_normal(mu=0.1, sigma=0.1, a=0, b=0.5)
        # transaction amount mean (amount will be sampled from an exponential distribution)
        self.transaction_amount_mean = exponential(1000)
        super().__init__()

    def make_session(self):
        """Make a session for a fraudster impersonating a legit customer."""

        # use random legit customer_id to make session
        self.customer_id = randint(n_customers)
        session = super().make_session()
        return session
